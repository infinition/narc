//! Depth reprojection baseline: backward warp from the nearest keyframes
//! with an occlusion test. Same inputs as NARC.
use crate::camera::{length, sub, Camera};
use crate::image::Rgb;
use crate::{is_sky, par_rows, Status};

/// Linear color.
pub struct Keyframe { pub index: usize, pub color: Rgb, pub depth: Vec<f32>, pub camera: Camera }

pub struct Reprojection {
    /// Nearest keyframes sampled per pixel.
    pub neighbours: usize,
    /// Relative occlusion tolerance.
    pub depth_tolerance: f64,
}
impl Default for Reprojection { fn default() -> Self { Self { neighbours: 2, depth_tolerance: 0.03 } } }

impl Reprojection {
    /// Linear color and per-pixel status. Holes stay black.
    pub fn reconstruct(&self, cam: &Camera, depth: &[f32], sky_ndc: Option<f32>, keys: &[Keyframe]) -> (Rgb, Vec<Status>) {
        let (w, h) = (cam.width, cam.height);
        let mut order: Vec<&Keyframe> = keys.iter().collect();
        order.sort_by(|a, b| length(sub(a.camera.position, cam.position)).total_cmp(&length(sub(b.camera.position, cam.position))).then(a.index.cmp(&b.index)));
        order.truncate(self.neighbours.max(1));
        let px = par_rows(h, |y| (0..w).map(|x| self.pixel(cam, depth[y * w + x], sky_ndc, x, y, &order)).collect());
        let mut img = Rgb::new(w, h);
        let mut status = Vec::with_capacity(w * h);
        for (i, (c, s)) in px.into_iter().enumerate() { img.data[i * 3..i * 3 + 3].copy_from_slice(&c); status.push(s); }
        (img, status)
    }

    fn pixel(&self, cam: &Camera, z: f32, sky_ndc: Option<f32>, x: usize, y: usize, keys: &[&Keyframe]) -> ([f32; 3], Status) {
        let (fx, fy) = (x as f64 + 0.5, y as f64 + 0.5);
        let point = if is_sky(z, sky_ndc) { None } else { cam.unproject(fx, fy, z as f64) };
        let mut acc = [0.0f32; 3];
        let mut wsum = 0.0f32;
        for k in keys {
            let kc = &k.camera;
            let (u, v) = match point {
                Some(p) => match kc.project(p) { Some((u, v, _)) => (u, v), None => continue },
                None => match kc.project_direction(cam.direction(fx, fy)) { Some(uv) => uv, None => continue },
            };
            if u < 0.0 || v < 0.0 || u >= kc.width as f64 || v >= kc.height as f64 { continue; }
            let (kx, ky) = (u as usize, v as usize);
            let kz = k.depth[ky * kc.width + kx];
            let visible = match point {
                None => is_sky(kz, sky_ndc),
                Some(p) => !is_sky(kz, sky_ndc) && kc.unproject(kx as f64 + 0.5, ky as f64 + 0.5, kz as f64).is_some_and(|q| {
                    let (dp, dq) = (length(sub(p, kc.position)), length(sub(q, kc.position)));
                    (dp - dq).abs() <= self.depth_tolerance * dp
                }),
            };
            if !visible { continue; }
            let wt = 1.0 / (length(sub(kc.position, cam.position)) as f32 + 1e-3);
            let c = k.color.bilinear(u, v);
            for j in 0..3 { acc[j] += c[j] * wt; }
            wsum += wt;
        }
        if wsum == 0.0 { return ([0.0; 3], Status::Hole); }
        (acc.map(|v| v / wsum), if point.is_some() { Status::Surface } else { Status::Sky })
    }
}
