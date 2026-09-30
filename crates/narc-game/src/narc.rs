//! Sparse appearance cache keyed by (voxel, octahedral view bin, normal class).
//! Sky pixels use a direction-only cache. `Interpolated` blends 8 voxels x 4 view
//! bins and falls back to the voxel mean when a view bin is empty.
use crate::camera::{normalize, sub, Camera, Vec3};
use crate::image::Rgb;
use crate::reproject::Keyframe;
use crate::{is_sky, par_rows, Status};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Lookup { Nearest, Interpolated }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NarcConfig {
    /// None: scale x median pixel footprint.
    pub voxel: Option<f64>,
    pub voxel_scale: f64,
    /// n x n octahedral bins.
    pub view_bins: u32,
    pub sky_bins: u32,
    /// Needs normals (level 2).
    pub use_normals: bool,
}
impl Default for NarcConfig { fn default() -> Self { Self { voxel: None, voxel_scale: 1.5, view_bins: 8, sky_bins: 64, use_normals: true } } }

type Key = (i32, i32, i32, u16, u8);
type Cell = (i32, i32, i32, u8);

pub struct NarcCache {
    pub voxel: f64,
    pub view_bins: u32,
    pub sky_bins: u32,
    pub use_normals: bool,
    entries: HashMap<Key, [f32; 4]>,
    cells: HashMap<Cell, [f32; 4]>,
    sky: HashMap<u16, [f32; 4]>,
}

/// Octahedral map of a unit vector to [0, 1]^2.
pub fn octahedral(d: Vec3) -> (f64, f64) {
    let s = d[0].abs() + d[1].abs() + d[2].abs();
    let (mut x, mut z) = (d[0] / s, d[2] / s);
    if d[1] < 0.0 { let (ax, az) = (x.abs(), z.abs()); x = (1.0 - az) * x.signum(); z = (1.0 - ax) * z.signum(); }
    (x * 0.5 + 0.5, z * 0.5 + 0.5)
}
fn bin(u: f64, v: f64, n: u32) -> u16 { let f = |t: f64| ((t * n as f64).floor() as i64).clamp(0, n as i64 - 1) as u32; (f(u) + n * f(v)) as u16 }
fn normal_class(n: Option<[f32; 3]>) -> u8 {
    let Some(n) = n else { return 0 };
    let (i, v) = n.iter().enumerate().max_by(|a, b| a.1.abs().total_cmp(&b.1.abs())).unwrap();
    1 + 2 * i as u8 + (*v < 0.0) as u8
}
fn add(m: &mut [f32; 4], c: [f32; 3], w: f32) { for k in 0..3 { m[k] += c[k] * w; } m[3] += w; }
fn mean(m: &[f32; 4]) -> [f32; 3] { [m[0] / m[3], m[1] / m[3], m[2] / m[3]] }

/// Median world distance between adjacent pixel centres.
pub fn pixel_footprint(keys: &[Keyframe], sky_ndc: Option<f32>) -> Option<f64> {
    let mut v = Vec::new();
    for k in keys {
        let c = &k.camera;
        for y in (0..c.height).step_by(7) { for x in (0..c.width - 1).step_by(7) {
            let z = k.depth[y * c.width + x];
            if is_sky(z, sky_ndc) { continue; }
            if let (Some(a), Some(b)) = (c.unproject(x as f64 + 0.5, y as f64 + 0.5, z as f64), c.unproject(x as f64 + 1.5, y as f64 + 0.5, z as f64)) { v.push(crate::camera::length(sub(b, a))); }
        } }
    }
    if v.is_empty() { return None; }
    v.sort_by(f64::total_cmp);
    Some(v[v.len() / 2])
}

impl NarcCache {
    pub fn bake(keys: &[Keyframe], normals: &[Option<Vec<[f32; 3]>>], sky_ndc: Option<f32>, cfg: &NarcConfig) -> anyhow::Result<Self> {
        let voxel = match cfg.voxel { Some(v) => v, None => pixel_footprint(keys, sky_ndc).ok_or_else(|| anyhow::anyhow!("no valid depth in keyframes"))? * cfg.voxel_scale };
        anyhow::ensure!(voxel.is_finite() && voxel > 0.0, "invalid voxel size {voxel}");
        anyhow::ensure!((1..=255).contains(&cfg.view_bins) && (1..=255).contains(&cfg.sky_bins), "bins must be in 1..=255");
        let use_normals = cfg.use_normals && normals.iter().all(|n| n.is_some());
        let mut c = Self { voxel, view_bins: cfg.view_bins, sky_bins: cfg.sky_bins, use_normals, entries: HashMap::new(), cells: HashMap::new(), sky: HashMap::new() };
        for (k, nrm) in keys.iter().zip(normals) {
            let cam = &k.camera;
            for y in 0..cam.height { for x in 0..cam.width {
                let i = y * cam.width + x;
                let (fx, fy) = (x as f64 + 0.5, y as f64 + 0.5);
                let color = k.color.px(x, y);
                let z = k.depth[i];
                let p = if is_sky(z, sky_ndc) { None } else { cam.unproject(fx, fy, z as f64) };
                match p {
                    None => { let (u, v) = octahedral(cam.direction(fx, fy)); add(c.sky.entry(bin(u, v, c.sky_bins)).or_default(), color, 1.0); }
                    Some(p) => {
                        let nc = if use_normals { normal_class(nrm.as_ref().map(|n| n[i])) } else { 0 };
                        let (u, v) = octahedral(normalize(sub(p, cam.position)));
                        let g = p.map(|a| (a / voxel).floor() as i32);
                        add(c.entries.entry((g[0], g[1], g[2], bin(u, v, c.view_bins), nc)).or_default(), color, 1.0);
                        add(c.cells.entry((g[0], g[1], g[2], nc)).or_default(), color, 1.0);
                    }
                }
            } }
        }
        Ok(c)
    }
    pub fn entries(&self) -> usize { self.entries.len() + self.cells.len() + self.sky.len() }
    /// 16-byte key + 16-byte payload per entry.
    pub fn bytes(&self) -> usize { self.entries() * 32 }

    pub fn reconstruct(&self, cam: &Camera, depth: &[f32], normals: Option<&[[f32; 3]]>, sky_ndc: Option<f32>, mode: Lookup) -> (Rgb, Vec<Status>) {
        let (w, h) = (cam.width, cam.height);
        let px = par_rows(h, |y| (0..w).map(|x| {
            let i = y * w + x;
            let nc = if self.use_normals { normal_class(normals.map(|n| n[i])) } else { 0 };
            self.pixel(cam, depth[i], nc, sky_ndc, x as f64 + 0.5, y as f64 + 0.5, mode)
        }).collect());
        let mut img = Rgb::new(w, h);
        let mut status = Vec::with_capacity(w * h);
        for (i, (c, s)) in px.into_iter().enumerate() { img.data[i * 3..i * 3 + 3].copy_from_slice(&c); status.push(s); }
        (img, status)
    }

    fn pixel(&self, cam: &Camera, z: f32, nc: u8, sky_ndc: Option<f32>, fx: f64, fy: f64, mode: Lookup) -> ([f32; 3], Status) {
        let point = if is_sky(z, sky_ndc) { None } else { cam.unproject(fx, fy, z as f64) };
        let Some(p) = point else {
            let (u, v) = octahedral(cam.direction(fx, fy));
            return match self.sky.get(&bin(u, v, self.sky_bins)) { Some(m) => (mean(m), Status::Sky), None => ([0.0; 3], Status::Hole) };
        };
        let (u, v) = octahedral(normalize(sub(p, cam.position)));
        match mode {
            Lookup::Nearest => {
                let g = p.map(|a| (a / self.voxel).floor() as i32);
                if let Some(m) = self.entries.get(&(g[0], g[1], g[2], bin(u, v, self.view_bins), nc)) { return (mean(m), Status::Surface); }
                match self.cells.get(&(g[0], g[1], g[2], nc)) { Some(m) => (mean(m), Status::ViewFallback), None => ([0.0; 3], Status::Hole) }
            }
            Lookup::Interpolated => {
                let t = p.map(|a| a / self.voxel - 0.5);
                let g0 = t.map(|a| a.floor());
                let n = self.view_bins as f64;
                let axis = |s: f64| { let q = s * n - 0.5; let i0 = q.floor().clamp(0.0, n - 1.0); (i0 as u32, (i0 as u32 + 1).min(self.view_bins - 1), (q - i0).clamp(0.0, 1.0)) };
                let (au, av) = (axis(u), axis(v));
                let (mut acc, mut wsum, mut fallback) = ([0.0f32; 3], 0.0f64, false);
                for corner in 0..32u32 {
                    let bit = |k: u32| corner >> k & 1 == 1;
                    let mut wt = 1.0;
                    let g: [i32; 3] = std::array::from_fn(|k| { let f = t[k] - g0[k]; wt *= if bit(k as u32) { f } else { 1.0 - f }; g0[k] as i32 + bit(k as u32) as i32 });
                    let (bu, wu) = if bit(3) { (au.1, au.2) } else { (au.0, 1.0 - au.2) };
                    let (bv, wv) = if bit(4) { (av.1, av.2) } else { (av.0, 1.0 - av.2) };
                    wt *= wu * wv;
                    if wt <= 0.0 { continue; }
                    let m = match self.entries.get(&(g[0], g[1], g[2], (bu + self.view_bins * bv) as u16, nc)) {
                        Some(m) => m,
                        None => match self.cells.get(&(g[0], g[1], g[2], nc)) { Some(m) => { fallback = true; m } None => continue },
                    };
                    let c = mean(m);
                    for k in 0..3 { acc[k] += c[k] * wt as f32; }
                    wsum += wt;
                }
                if wsum <= 1e-9 { return ([0.0; 3], Status::Hole); }
                (acc.map(|a| a / wsum as f32), if fallback { Status::ViewFallback } else { Status::Surface })
            }
        }
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(b"NARCGAME")?;
        f.write_all(&1u32.to_le_bytes())?;
        f.write_all(&self.voxel.to_le_bytes())?;
        for v in [self.view_bins, self.sky_bins, self.use_normals as u32, self.entries.len() as u32, self.cells.len() as u32, self.sky.len() as u32] { f.write_all(&v.to_le_bytes())?; }
        let payload = |f: &mut std::io::BufWriter<std::fs::File>, m: &[f32; 4]| -> std::io::Result<()> { for v in m { f.write_all(&v.to_le_bytes())?; } Ok(()) };
        let mut e: Vec<_> = self.entries.iter().collect(); e.sort_by_key(|x| *x.0);
        for ((x, y, z, b, n), m) in e { for v in [*x, *y, *z] { f.write_all(&v.to_le_bytes())?; } f.write_all(&b.to_le_bytes())?; f.write_all(&[*n, 0])?; payload(&mut f, m)?; }
        let mut c: Vec<_> = self.cells.iter().collect(); c.sort_by_key(|x| *x.0);
        for ((x, y, z, n), m) in c { for v in [*x, *y, *z] { f.write_all(&v.to_le_bytes())?; } f.write_all(&[*n, 0, 0, 0])?; payload(&mut f, m)?; }
        let mut s: Vec<_> = self.sky.iter().collect(); s.sort_by_key(|x| *x.0);
        for (b, m) in s { f.write_all(&b.to_le_bytes())?; payload(&mut f, m)?; }
        f.flush()?;
        Ok(())
    }
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 8]; f.read_exact(&mut magic)?; anyhow::ensure!(&magic == b"NARCGAME", "not a NARC game cache");
        let mut b4 = [0u8; 4]; let mut b8 = [0u8; 8]; let mut b2 = [0u8; 2];
        let mut u32r = |f: &mut std::io::BufReader<std::fs::File>| -> std::io::Result<u32> { f.read_exact(&mut b4)?; Ok(u32::from_le_bytes(b4)) };
        anyhow::ensure!(u32r(&mut f)? == 1, "unsupported cache version");
        f.read_exact(&mut b8)?; let voxel = f64::from_le_bytes(b8);
        let (view_bins, sky_bins, use_normals, ne, nc, ns) = (u32r(&mut f)?, u32r(&mut f)?, u32r(&mut f)? == 1, u32r(&mut f)?, u32r(&mut f)?, u32r(&mut f)?);
        let i32r = |f: &mut std::io::BufReader<std::fs::File>| -> std::io::Result<i32> { let mut b = [0u8; 4]; f.read_exact(&mut b)?; Ok(i32::from_le_bytes(b)) };
        let payload = |f: &mut std::io::BufReader<std::fs::File>| -> std::io::Result<[f32; 4]> { let mut m = [0f32; 4]; for v in &mut m { let mut b = [0u8; 4]; f.read_exact(&mut b)?; *v = f32::from_le_bytes(b); } Ok(m) };
        let mut c = Self { voxel, view_bins, sky_bins, use_normals, entries: HashMap::new(), cells: HashMap::new(), sky: HashMap::new() };
        for _ in 0..ne { let (x, y, z) = (i32r(&mut f)?, i32r(&mut f)?, i32r(&mut f)?); f.read_exact(&mut b2)?; let b = u16::from_le_bytes(b2); f.read_exact(&mut b2)?; c.entries.insert((x, y, z, b, b2[0]), payload(&mut f)?); }
        for _ in 0..nc { let (x, y, z) = (i32r(&mut f)?, i32r(&mut f)?, i32r(&mut f)?); f.read_exact(&mut b4)?; c.cells.insert((x, y, z, b4[0]), payload(&mut f)?); }
        for _ in 0..ns { f.read_exact(&mut b2)?; c.sky.insert(u16::from_le_bytes(b2), payload(&mut f)?); }
        Ok(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn octahedral_covers_the_sphere() {
        for d in [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [-0.3, -0.8, 0.52], [0.577, 0.577, -0.577]] {
            let (u, v) = octahedral(normalize(d));
            assert!((0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v));
            assert!(bin(u, v, 8) < 64);
        }
        assert_ne!(octahedral([0.0, 1.0, 0.0]), octahedral([0.0, -1.0, 0.0]));
    }
    #[test]
    fn normal_classes_are_distinct() {
        let c: Vec<u8> = [[1.0, 0.1, 0.0], [-1.0, 0.0, 0.2], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.1, 0.0, 1.0], [0.0, 0.0, -1.0]].iter().map(|n| normal_class(Some(*n))).collect();
        assert_eq!(c, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(normal_class(None), 0);
    }
}
