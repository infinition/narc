//! import, validate, bake, evaluate.
use crate::capture::{DataLevel, FrameSource, SceneClass};
use crate::image::{self, Rgb};
use crate::metrics::{self, Quality};
use crate::narc::{Lookup, NarcCache, NarcConfig};
use crate::reproject::{Keyframe, Reprojection};
use crate::{fill_holes, is_sky, Status};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImportReport {
    pub name: String,
    pub source: String,
    pub resolution: String,
    pub frames: usize,
    pub level: DataLevel,
    pub level_description: String,
    pub scene_class: SceneClass,
    pub frames_with_camera: usize,
    pub frames_with_depth: usize,
    pub sky_fraction: f64,
    pub depth_ndc_min: f64,
    pub depth_ndc_max: f64,
    pub camera_path_length: f64,
    pub geometry_usable: bool,
    /// Frame 1 reprojected from frame 0. Collapses on a wrong matrix convention.
    pub pair_check_psnr_db: Option<f64>,
    pub pair_check_coverage: Option<f64>,
    pub issues: Vec<String>,
}

pub fn import(src: &dyn FrameSource) -> anyhow::Result<ImportReport> {
    let m = src.manifest();
    let level = src.level()?;
    let (mut with_cam, mut with_depth, mut sky, mut total) = (0, 0, 0usize, 0usize);
    let (mut zmin, mut zmax) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut cams = Vec::new();
    let mut issues = Vec::new();
    for i in 0..src.len() {
        let meta = src.metadata(i)?;
        if meta.index != i { issues.push(format!("frame {i}: metadata.index is {}", meta.index)); }
        src.color(i)?;
        match src.camera(i) { Ok(Some(c)) => { with_cam += 1; cams.push(c); } Ok(None) => issues.push(format!("frame {i}: no camera")), Err(e) => issues.push(format!("frame {i}: camera: {e}")) }
        if let Some(d) = src.depth(i)? {
            with_depth += 1;
            for &z in &d { total += 1; if is_sky(z, m.depth.sky_ndc) { sky += 1; } else { zmin = zmin.min(z as f64); zmax = zmax.max(z as f64); } }
        }
    }
    let geometry_usable = level >= DataLevel::L1Depth && with_cam == src.len();
    if !geometry_usable { issues.push("geometry unusable: reprojection and NARC need depth and camera on every frame (LEVEL 1+)".into()); }
    let (mut pair_psnr, mut pair_cov) = (None, None);
    if geometry_usable && src.len() >= 2 {
        let (keys, _) = load_keyframes(src, &[0])?;
        let (lin, status) = Reprojection::default().reconstruct(&src.camera(1)?.expect("checked"), &src.depth(1)?.expect("checked"), m.depth.sky_ndc, &keys);
        let mut rec = lin.to_srgb();
        let covered: Vec<bool> = status.iter().map(|s| s.covered()).collect();
        fill_holes(&mut rec, &status);
        let q = metrics::compare(&src.color(1)?, &rec, &covered);
        if q.coverage < 0.5 { issues.push(format!("camera check: only {:.0}% of frame 1 reprojects from frame 0; check world_to_clip convention and depth", q.coverage * 100.0)); }
        (pair_psnr, pair_cov) = (q.psnr_covered_db, Some(q.coverage));
    }
    Ok(ImportReport {
        name: m.name.clone(), source: m.source.clone(), resolution: format!("{}x{}", m.width, m.height), frames: src.len(), level, level_description: level.describe().into(),
        scene_class: m.scene_class, frames_with_camera: with_cam, frames_with_depth: with_depth, sky_fraction: if total == 0 { 0.0 } else { sky as f64 / total as f64 },
        depth_ndc_min: zmin, depth_ndc_max: zmax, camera_path_length: crate::mock::path_length(&cams), geometry_usable, pair_check_psnr_db: pair_psnr, pair_check_coverage: pair_cov, issues,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BakeReport {
    pub capture: String,
    pub name: String,
    pub level: DataLevel,
    pub every: usize,
    pub keyframes: Vec<usize>,
    pub narc: NarcConfig,
    pub voxel: f64,
    pub normals_in_key: bool,
    pub cache_entries: usize,
    pub cache_bytes: usize,
    /// Keyframes as RGBA8 + D32, kept by the reprojection baseline.
    pub keyframe_store_bytes: usize,
    pub bake_cpu_ms: f64,
}

pub fn keyframe_indices(len: usize, every: usize) -> Vec<usize> { (0..len).step_by(every.max(1)).collect() }

fn load_keyframes(src: &dyn FrameSource, idx: &[usize]) -> anyhow::Result<(Vec<Keyframe>, Vec<Option<Vec<[f32; 3]>>>)> {
    let mut keys = Vec::new();
    let mut normals = Vec::new();
    for &i in idx {
        let depth = src.depth(i)?.ok_or_else(|| anyhow::anyhow!("keyframe {i}: depth missing"))?;
        let camera = src.camera(i)?.ok_or_else(|| anyhow::anyhow!("keyframe {i}: camera missing"))?;
        keys.push(Keyframe { index: i, color: src.color(i)?.to_linear(), depth, camera });
        normals.push(src.normals(i)?);
    }
    Ok((keys, normals))
}

pub fn bake(src: &dyn FrameSource, capture: &Path, out: &Path, every: usize, cfg: &NarcConfig) -> anyhow::Result<BakeReport> {
    let level = src.level()?;
    anyhow::ensure!(level >= DataLevel::L1Depth, "bake needs LEVEL 1 (color + depth + camera); capture is {}", level.describe());
    anyhow::ensure!(every >= 2, "--every must be at least 2, otherwise no frame is unseen");
    let idx = keyframe_indices(src.len(), every);
    let (keys, normals) = load_keyframes(src, &idx)?;
    let t = Instant::now();
    let cache = NarcCache::bake(&keys, &normals, src.manifest().depth.sky_ndc, cfg)?;
    let bake_cpu_ms = t.elapsed().as_secs_f64() * 1e3;
    std::fs::create_dir_all(out)?;
    cache.save(&out.join("narc_cache.bin"))?;
    let (w, h) = src.size();
    let report = BakeReport {
        capture: capture.display().to_string(), name: src.manifest().name.clone(), level, every, keyframes: idx.clone(), narc: cfg.clone(), voxel: cache.voxel,
        normals_in_key: cache.use_normals, cache_entries: cache.entries(), cache_bytes: cache.bytes(), keyframe_store_bytes: idx.len() * w * h * 8, bake_cpu_ms,
    };
    std::fs::write(out.join("bake.json"), serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameRow {
    pub frame: usize,
    pub baked: bool,
    /// Distance in frames to the nearest keyframe.
    pub keyframe_gap: usize,
    pub method: String,
    pub psnr_db: Option<f64>,
    pub psnr_covered_db: Option<f64>,
    pub ssim: f64,
    pub rmse: f64,
    pub max_abs_error: f64,
    pub coverage: f64,
    pub view_fallback: f64,
    pub temporal_rmse: Option<f64>,
    pub cpu_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MethodSummary {
    pub method: String,
    pub unseen_frames: usize,
    /// Pooled over unseen frames.
    pub psnr_db: Option<f64>,
    pub worst_frame_psnr_db: Option<f64>,
    pub ssim_mean: f64,
    pub rmse: f64,
    pub max_abs_error: f64,
    pub coverage_mean: f64,
    pub temporal_rmse_mean: Option<f64>,
    pub cpu_ms_median: f64,
    pub memory_bytes: usize,
    /// Keyframes only, sanity check.
    pub baked_psnr_db: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalReport {
    pub name: String,
    pub resolution: String,
    pub level: DataLevel,
    pub scene_class: SceneClass,
    pub keyframes: Vec<usize>,
    pub unseen_frames: Vec<usize>,
    pub summaries: Vec<MethodSummary>,
    pub notes: Vec<String>,
}

pub const METHODS: [&str; 3] = ["reprojection", "narc-nearest", "narc-interpolated"];

pub fn evaluate(src: &dyn FrameSource, bake_dir: &Path, out: &Path, images: usize) -> anyhow::Result<EvalReport> {
    let bake: BakeReport = serde_json::from_slice(&std::fs::read(bake_dir.join("bake.json"))?)?;
    anyhow::ensure!(bake.name == src.manifest().name, "bake.json belongs to capture {}, not {}", bake.name, src.manifest().name);
    let cache = NarcCache::load(&bake_dir.join("narc_cache.bin"))?;
    let (keys, _) = load_keyframes(src, &bake.keyframes)?;
    let sky = src.manifest().depth.sky_ndc;
    let (w, h) = src.size();
    let unseen: Vec<usize> = (0..src.len()).filter(|i| !bake.keyframes.contains(i)).collect();
    anyhow::ensure!(!unseen.is_empty(), "every frame is a keyframe: nothing unseen to evaluate");
    let picks: Vec<usize> = if images == 0 { vec![] } else {
        // Unseen frames farthest from any keyframe.
        let mut far: Vec<usize> = unseen.iter().copied().filter(|&i| gap(i, &bake.keyframes) == unseen.iter().map(|&j| gap(j, &bake.keyframes)).max().unwrap_or(0)).collect();
        let step = far.len().div_ceil(images).max(1);
        far = far.into_iter().step_by(step).take(images).collect();
        far
    };
    let reproj = Reprojection::default();
    let memory = [bake.keyframe_store_bytes, bake.cache_bytes, bake.cache_bytes];
    let mut rows: Vec<FrameRow> = Vec::new();
    let mut prev: Option<(Rgb, Vec<Rgb>)> = None;
    std::fs::create_dir_all(out)?;
    for i in 0..src.len() {
        let orig = src.color(i)?;
        let depth = src.depth(i)?.ok_or_else(|| anyhow::anyhow!("frame {i}: depth missing"))?;
        let cam = src.camera(i)?.ok_or_else(|| anyhow::anyhow!("frame {i}: camera missing"))?;
        let normals = src.normals(i)?;
        let mut recons = Vec::new();
        for m in METHODS {
            let t = Instant::now();
            let (lin, status) = match m {
                "reprojection" => reproj.reconstruct(&cam, &depth, sky, &keys),
                "narc-nearest" => cache.reconstruct(&cam, &depth, normals.as_deref(), sky, Lookup::Nearest),
                _ => cache.reconstruct(&cam, &depth, normals.as_deref(), sky, Lookup::Interpolated),
            };
            let cpu_ms = t.elapsed().as_secs_f64() * 1e3;
            let mut rec = lin.to_srgb();
            let covered: Vec<bool> = status.iter().map(|s| s.covered()).collect();
            let fallback = status.iter().filter(|s| **s == Status::ViewFallback).count() as f64 / (w * h) as f64;
            let raw = rec.clone();
            fill_holes(&mut rec, &status);
            let q: Quality = metrics::compare(&orig, &rec, &covered);
            let temporal = prev.as_ref().map(|(po, pr)| metrics::temporal_rmse(po, &orig, &pr[recons.len()], &rec));
            rows.push(FrameRow { frame: i, baked: bake.keyframes.contains(&i), keyframe_gap: gap(i, &bake.keyframes), method: m.into(), psnr_db: q.psnr_db, psnr_covered_db: q.psnr_covered_db, ssim: q.ssim, rmse: q.rmse, max_abs_error: q.max_abs_error, coverage: q.coverage, view_fallback: fallback, temporal_rmse: temporal, cpu_ms });
            if picks.contains(&i) {
                let d = out.join(format!("frame_{i:06}"));
                std::fs::create_dir_all(&d)?;
                image::save_png(&d.join(format!("{m}.png")), &rec)?;
                image::save_png(&d.join(format!("{m}_raw_holes_black.png")), &raw)?;
                image::save_png(&d.join(format!("heatmap_{m}.png")), &image::heatmap(&orig, &rec, 0.25))?;
                image::save_png(&d.join(format!("holes_{m}.png")), &image::mask(w, h, |p| !covered[p]))?;
                image::save_png(&d.join("original.png"), &orig)?;
            }
            recons.push(rec);
        }
        prev = Some((orig, recons));
        eprintln!("frame {i}/{} {}", src.len(), if bake.keyframes.contains(&i) { "(keyframe)" } else { "" });
    }
    let summaries = METHODS.iter().enumerate().map(|(k, m)| summarize(m, &rows, memory[k])).collect();
    let report = EvalReport {
        name: src.manifest().name.clone(), resolution: format!("{w}x{h}"), level: src.level()?, scene_class: src.manifest().scene_class, keyframes: bake.keyframes.clone(), unseen_frames: unseen,
        summaries,
        notes: vec![
            "All methods receive the same inputs: target depth + camera, keyframe color + depth + camera. No method sees the original color of an unseen frame.".into(),
            "Summaries cover unseen frames only; baked_psnr_db is a sanity check on keyframes.".into(),
            "Holes (no data) are filled by the nearest covered pixel on the row before whole-frame metrics; coverage and psnr_covered_db report them separately.".into(),
            "cpu_ms is a single-machine CPU reference timing, not GPU time. GPU ports are a next step.".into(),
            "memory_bytes: reprojection keeps every keyframe (RGBA8 + D32); NARC keeps only its cache (32 B per entry).".into(),
            "temporal_rmse: RMSE of (R_t - R_t-1) - (O_t - O_t-1) over consecutive frames.".into(),
        ],
    };
    std::fs::write(out.join("metrics.json"), serde_json::to_vec_pretty(&report)?)?;
    let mut csv = String::from("frame,baked,keyframe_gap,method,psnr_db,psnr_covered_db,ssim,rmse,max_abs_error,coverage,view_fallback,temporal_rmse,cpu_ms\n");
    let o = |v: Option<f64>| v.map_or(String::new(), |x| format!("{x:.4}"));
    for r in &rows { csv.push_str(&format!("{},{},{},{},{},{},{:.5},{:.6},{:.5},{:.5},{:.5},{},{:.2}\n", r.frame, r.baked, r.keyframe_gap, r.method, o(r.psnr_db), o(r.psnr_covered_db), r.ssim, r.rmse, r.max_abs_error, r.coverage, r.view_fallback, o(r.temporal_rmse), r.cpu_ms)); }
    std::fs::write(out.join("metrics.csv"), csv)?;
    Ok(report)
}

fn gap(i: usize, keys: &[usize]) -> usize { keys.iter().map(|&k| i.abs_diff(k)).min().unwrap_or(usize::MAX) }

fn summarize(method: &str, rows: &[FrameRow], memory_bytes: usize) -> MethodSummary {
    let sel = |baked: bool| rows.iter().filter(move |r| r.method == method && r.baked == baked);
    let pooled = |baked: bool| { let v: Vec<f64> = sel(baked).map(|r| r.rmse * r.rmse).collect(); if v.is_empty() { None } else { metrics::psnr(v.iter().sum::<f64>() / v.len() as f64) } };
    let u: Vec<&FrameRow> = sel(false).collect();
    let n = u.len().max(1) as f64;
    let mut ms: Vec<f64> = u.iter().map(|r| r.cpu_ms).collect();
    ms.sort_by(f64::total_cmp);
    let temporal: Vec<f64> = u.iter().filter_map(|r| r.temporal_rmse).collect();
    MethodSummary {
        method: method.into(), unseen_frames: u.len(), psnr_db: pooled(false),
        worst_frame_psnr_db: u.iter().filter_map(|r| r.psnr_db).fold(None, |a: Option<f64>, b| Some(a.map_or(b, |a| a.min(b)))),
        ssim_mean: u.iter().map(|r| r.ssim).sum::<f64>() / n, rmse: (u.iter().map(|r| r.rmse * r.rmse).sum::<f64>() / n).sqrt(),
        max_abs_error: u.iter().map(|r| r.max_abs_error).fold(0.0, f64::max), coverage_mean: u.iter().map(|r| r.coverage).sum::<f64>() / n,
        temporal_rmse_mean: if temporal.is_empty() { None } else { Some(temporal.iter().sum::<f64>() / temporal.len() as f64) },
        cpu_ms_median: ms.get(ms.len() / 2).copied().unwrap_or(f64::NAN), memory_bytes, baked_psnr_db: pooled(true),
    }
}

pub fn print_summary(r: &EvalReport) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(s, "NARC capture evaluation - {} - {} - {} - {:?}", r.name, r.resolution, r.level.describe(), r.scene_class);
    let _ = writeln!(s, "keyframes {} ({:?}...), unseen frames evaluated {}", r.keyframes.len(), &r.keyframes[..r.keyframes.len().min(6)], r.unseen_frames.len());
    let _ = writeln!(s, "{:<18} {:>9} {:>10} {:>7} {:>8} {:>9} {:>10} {:>9} {:>10} {:>10}", "method", "PSNR dB", "worst dB", "SSIM", "RMSE", "coverage", "temporal", "CPU ms", "memory MB", "baked dB");
    let db = |v: Option<f64>| v.map_or("identical".into(), |x| format!("{x:.2}"));
    for m in &r.summaries {
        let _ = writeln!(s, "{:<18} {:>9} {:>10} {:>7.4} {:>8.5} {:>8.1}% {:>10} {:>9.1} {:>10.1} {:>10}", m.method, db(m.psnr_db), db(m.worst_frame_psnr_db), m.ssim_mean, m.rmse, m.coverage_mean * 100.0,
            m.temporal_rmse_mean.map_or("-".into(), |t| format!("{t:.5}")), m.cpu_ms_median, m.memory_bytes as f64 / 1048576.0, db(m.baked_psnr_db));
    }
    s
}


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Check { pub name: String, pub ok: bool, pub detail: String }
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Validation { pub import: ImportReport, pub checks: Vec<Check>, pub pair_psnr_db: Vec<Option<f64>>, pub ready: bool }

/// Pre-bake checklist. Loose thresholds: they only catch broken conventions.
pub fn validate(src: &dyn FrameSource, pairs: usize) -> anyhow::Result<Validation> {
    let imp = import(src)?;
    let m = src.manifest();
    let mut checks = Vec::new();
    let mut add = |name: &str, ok: bool, detail: String| checks.push(Check { name: name.into(), ok, detail });
    let color_ok = (0..src.len()).all(|i| src.color(i).is_ok());
    add("color", color_ok, format!("{} frames, {}", src.len(), imp.resolution));
    add("depth", imp.frames_with_depth == src.len(), format!("{}/{} frames with depth.exr", imp.frames_with_depth, src.len()));
    let range_ok = imp.depth_ndc_min.is_finite() && imp.depth_ndc_min >= -1e-6 && imp.depth_ndc_max <= 1.0 + 1e-6 && imp.depth_ndc_max > imp.depth_ndc_min && imp.sky_fraction < 0.98;
    add("depth range", range_ok, format!("NDC [{:.6}, {:.6}], sky {:.1}% (sky_ndc {:?})", imp.depth_ndc_min, imp.depth_ndc_max, imp.sky_fraction * 100.0, m.depth.sky_ndc));
    add("camera", imp.frames_with_camera == src.len(), format!("{}/{} frames with an invertible world_to_clip", imp.frames_with_camera, src.len()));
    let moving = imp.camera_path_length > 0.0;
    add("camera motion", moving, format!("camera path length {:.4} world units", imp.camera_path_length));
    let mut pair_psnr = Vec::new();
    let mut matrix_ok = imp.geometry_usable && src.len() >= 2;
    if matrix_ok {
        let n = pairs.clamp(1, src.len() - 1);
        for k in 0..n {
            let i = k * (src.len() - 1) / n;
            let (keys, _) = load_keyframes(src, &[i])?;
            let (lin, status) = Reprojection::default().reconstruct(&src.camera(i + 1)?.expect("checked"), &src.depth(i + 1)?.expect("checked"), m.depth.sky_ndc, &keys);
            let mut rec = lin.to_srgb();
            let covered: Vec<bool> = status.iter().map(|s| s.covered()).collect();
            fill_holes(&mut rec, &status);
            let q = metrics::compare(&src.color(i + 1)?, &rec, &covered);
            matrix_ok &= q.coverage >= 0.5 && q.psnr_covered_db.map_or(true, |p| p >= 20.0);
            pair_psnr.push(q.psnr_covered_db);
        }
    }
    let shown: Vec<String> = pair_psnr.iter().map(|p| p.map_or("identical".into(), |v| format!("{v:.1}"))).collect();
    add("matrix", matrix_ok, format!("pairwise reprojection over covered pixels (dB): [{}]; needs coverage >= 50% and >= 20 dB", shown.join(", ")));
    add("frames", src.len() >= 16, format!("{} frames (>= 16 recommended to keep unseen frames after baking)", src.len()));
    let ready = checks.iter().all(|c| c.ok);
    Ok(Validation { import: imp, checks, pair_psnr_db: pair_psnr, ready })
}
