//! Deterministic CPU ray-traced capture in the on-disk format, for tests.
use crate::camera::{self, dot, length, normalize, sub, Camera, CameraMeta, Vec3};
use crate::capture::{self, DepthSpec, FrameMeta, Manifest, SceneClass};
use crate::image::{linear_to_srgb, Rgb};
use crate::par_rows;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum MockKind {
    /// Camera moves, scene and light fixed.
    Static,
    /// Camera moves and the sun rotates.
    Lighting,
    /// Camera and one object move.
    Dynamic,
}
impl std::str::FromStr for MockKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s { "static" => Ok(Self::Static), "lighting" => Ok(Self::Lighting), "dynamic" => Ok(Self::Dynamic), _ => anyhow::bail!("mock kind must be static, lighting or dynamic") }
    }
}

struct Sphere { c: Vec3, r: f64, albedo: [f64; 3], gloss: f64 }
struct Scene { spheres: Vec<Sphere>, sun: Vec3 }
const NEAR: f64 = 0.1;
const FOV: f64 = 60.0;

fn scene(kind: MockKind, t: f64) -> Scene {
    let mut spheres = vec![
        Sphere { c: [0.0, 1.0, 0.0], r: 1.0, albedo: [0.75, 0.18, 0.12], gloss: 0.7 },
        Sphere { c: [2.4, 0.6, 1.4], r: 0.6, albedo: [0.15, 0.3, 0.8], gloss: 0.0 },
        Sphere { c: [-2.2, 0.8, -1.2], r: 0.8, albedo: [0.2, 0.65, 0.25], gloss: 0.15 },
    ];
    if kind == MockKind::Dynamic { spheres.push(Sphere { c: [-3.5 + 7.0 * t, 0.5, -2.6], r: 0.5, albedo: [0.9, 0.8, 0.1], gloss: 0.3 }); }
    let (az, el) = if kind == MockKind::Lighting { ((30.0 + 100.0 * t).to_radians(), (55.0 - 30.0 * t).to_radians()) } else { (30f64.to_radians(), 55f64.to_radians()) };
    Scene { spheres, sun: [el.cos() * az.cos(), el.sin(), el.cos() * az.sin()] }
}

enum Hit { Sky, Surface { t: f64, n: Vec3, albedo: [f64; 3], gloss: f64 } }
fn trace(s: &Scene, o: Vec3, d: Vec3) -> Hit {
    let mut best: Option<(f64, Vec3, [f64; 3], f64)> = None;
    // Finite ground avoids an aliased horizon.
    let t = -o[1] / d[1];
    let p = [o[0] + t * d[0], 0.0, o[2] + t * d[2]];
    if d[1] < -1e-9 && p[0] * p[0] + p[2] * p[2] < 14.0 * 14.0 {
        let checker = ((p[0].floor() + p[2].floor()) as i64).rem_euclid(2) as f64;
        let detail = 0.08 * (7.0 * p[0]).sin() * (5.0 * p[2]).sin();
        let a = 0.35 + 0.25 * checker + detail;
        best = Some((t, [0.0, 1.0, 0.0], [a * 0.95, a * 0.9, a * 0.8], 0.0));
    }
    for sp in &s.spheres {
        let oc = sub(o, sp.c);
        let b = dot(oc, d);
        let disc = b * b - (dot(oc, oc) - sp.r * sp.r);
        if disc < 0.0 { continue; }
        let t = -b - disc.sqrt();
        if t > 1e-6 && best.as_ref().map_or(true, |x| t < x.0) {
            let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
            let n = normalize(sub(p, sp.c));
            let stripe = if sp.gloss > 0.1 && sp.gloss < 0.2 { 0.85 + 0.15 * (12.0 * n[1]).sin() } else { 1.0 };
            best = Some((t, n, sp.albedo.map(|v| v * stripe), sp.gloss));
        }
    }
    match best { None => Hit::Sky, Some((t, n, albedo, gloss)) => Hit::Surface { t, n, albedo, gloss } }
}
fn sky(d: Vec3, sun: Vec3) -> [f64; 3] {
    let h = d[1].max(0.0);
    let s = dot(d, sun).max(0.0).powf(400.0) * 4.0;
    [0.55 - 0.3 * h + s, 0.7 - 0.25 * h + s, 0.95 + s * 0.8]
}
fn shade(s: &Scene, o: Vec3, d: Vec3) -> [f64; 3] {
    match trace(s, o, d) {
        Hit::Sky => sky(d, s.sun),
        Hit::Surface { t, n, albedo, gloss } => {
            let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
            let q = [p[0] + n[0] * 1e-4, p[1] + n[1] * 1e-4, p[2] + n[2] * 1e-4];
            let lit = if matches!(trace(s, q, s.sun), Hit::Sky) { 1.0 } else { 0.0 };
            let diffuse = dot(n, s.sun).max(0.0) * lit;
            let h = normalize(sub(s.sun, d));
            let spec = gloss * dot(n, h).max(0.0).powf(80.0) * lit * 1.5;
            std::array::from_fn(|k| albedo[k] * (0.22 + 0.85 * diffuse) + spec)
        }
    }
}

pub fn camera_at(t: f64, width: usize, height: usize) -> anyhow::Result<Camera> {
    let a = (-35.0 + 70.0 * t).to_radians();
    let eye = [7.0 * a.sin(), 2.2 + 0.4 * t, -7.0 * a.cos()];
    let m = camera::look_at_reversed_infinite(eye, [0.0, 0.7, 0.0], [0.0, 1.0, 0.0], FOV, width as f64 / height as f64, NEAR);
    Camera::new(&CameraMeta { world_to_clip: m, position: None }, width, height)
}

/// sRGB color (3x3 supersampled), NDC depth, world normals.
pub fn render(kind: MockKind, t: f64, width: usize, height: usize) -> anyhow::Result<(Rgb, Vec<f32>, Vec<f32>, Camera)> {
    let cam = camera_at(t, width, height)?;
    let s = scene(kind, t);
    let px: Vec<([f32; 3], f32, [f32; 3])> = par_rows(height, |y| {
        (0..width).map(|x| {
            let mut c = [0.0; 3];
            for (sx, sy) in (0..9).map(|s| ((s % 3) as f64 / 3.0 + 1.0 / 6.0, (s / 3) as f64 / 3.0 + 1.0 / 6.0)) {
                let v = shade(&s, cam.position, cam.direction(x as f64 + sx, y as f64 + sy));
                for k in 0..3 { c[k] += v[k] / 9.0; }
            }
            let d = cam.direction(x as f64 + 0.5, y as f64 + 0.5);
            let (z, n) = match trace(&s, cam.position, d) {
                Hit::Sky => (0.0, [0.0; 3]),
                Hit::Surface { t, n, .. } => {
                    let p = [cam.position[0] + t * d[0], cam.position[1] + t * d[1], cam.position[2] + t * d[2]];
                    (cam.project(p).map_or(0.0, |q| q.2 as f32), n.map(|v| v as f32))
                }
            };
            (c.map(|v| linear_to_srgb(v as f32)), z, n)
        }).collect()
    });
    let mut color = Rgb::new(width, height);
    for (i, p) in px.iter().enumerate() { color.data[i * 3..i * 3 + 3].copy_from_slice(&p.0); }
    Ok((color, px.iter().map(|p| p.1).collect(), px.iter().flat_map(|p| p.2).collect(), cam))
}

pub fn generate(root: &Path, kind: MockKind, width: usize, height: usize, frames: usize) -> anyhow::Result<Manifest> {
    anyhow::ensure!(frames >= 2 && width >= 16 && height >= 16, "mock needs at least 2 frames and 16x16 pixels");
    std::fs::create_dir_all(root)?;
    let mut names = Vec::new();
    for i in 0..frames {
        let t = i as f64 / (frames - 1) as f64;
        let (color, depth, normals, cam) = render(kind, t, width, height)?;
        let tags = match kind { MockKind::Dynamic => vec!["dynamic-object".to_string()], MockKind::Lighting => vec!["light-change".to_string()], MockKind::Static => vec![] };
        let s = scene(kind, t);
        let meta = FrameMeta { index: i, time_s: i as f64 / 30.0, camera: Some(CameraMeta { world_to_clip: cam.world_to_clip, position: Some(cam.position) }), tags, extra: serde_json::json!({ "sun_direction": s.sun }) };
        let name = capture::frame_dir_name(i);
        capture::write_frame(root, &name, &color, Some(&depth), Some(&normals), &meta)?;
        names.push(name);
    }
    let manifest = Manifest {
        format: capture::FORMAT.into(), version: capture::VERSION, name: format!("mock-{kind:?}").to_lowercase(), source: "mock".into(),
        width, height, depth: DepthSpec { kind: "ndc".into(), sky_ndc: Some(0.0) },
        scene_class: if kind == MockKind::Dynamic { SceneClass::DynamicUnsafe } else { SceneClass::StaticCacheable },
        frames: names,
        notes: vec![format!("Deterministic CPU ray-traced mock ({kind:?}), reversed-Z infinite projection, finite ground disc, sRGB color with 3x3 supersampling, depth and normals at pixel centres.")],
    };
    std::fs::write(root.join("capture.json"), serde_json::to_vec_pretty(&manifest)?)?;
    Ok(manifest)
}

/// Total distance travelled by the camera centre.
pub fn path_length(cams: &[Camera]) -> f64 { cams.windows(2).map(|w| length(sub(w[1].position, w[0].position))).sum() }
