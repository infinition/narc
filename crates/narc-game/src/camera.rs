//! Camera = `world_to_clip` (row-major, clip = M * [x y z 1]) + raw NDC depth.
//! Convention-free unprojection (handedness, reversed Z, infinite far plane).
//! Direct3D image convention: NDC y up, row 0 at the top.
use serde::{Deserialize, Serialize};

pub type Vec3 = [f64; 3];
pub type Mat4 = [[f64; 4]; 4];

pub fn sub(a: Vec3, b: Vec3) -> Vec3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
pub fn dot(a: Vec3, b: Vec3) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
pub fn length(a: Vec3) -> f64 { dot(a, a).sqrt() }
pub fn normalize(a: Vec3) -> Vec3 { let l = length(a).max(1e-300); [a[0] / l, a[1] / l, a[2] / l] }
pub fn cross(a: Vec3, b: Vec3) -> Vec3 { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }

pub fn mul4(m: &Mat4, v: [f64; 4]) -> [f64; 4] {
    std::array::from_fn(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2] + m[r][3] * v[3])
}
pub fn matmul(a: &Mat4, b: &Mat4) -> Mat4 {
    std::array::from_fn(|r| std::array::from_fn(|c| (0..4).map(|k| a[r][k] * b[k][c]).sum()))
}
/// General 4x4 inverse (cofactor expansion). Returns None for singular matrices.
pub fn invert(m: &Mat4) -> Option<Mat4> {
    let a: Vec<f64> = m.iter().flatten().copied().collect();
    let mut inv = [0.0f64; 16];
    inv[0] = a[5]*a[10]*a[15]-a[5]*a[11]*a[14]-a[9]*a[6]*a[15]+a[9]*a[7]*a[14]+a[13]*a[6]*a[11]-a[13]*a[7]*a[10];
    inv[4] = -a[4]*a[10]*a[15]+a[4]*a[11]*a[14]+a[8]*a[6]*a[15]-a[8]*a[7]*a[14]-a[12]*a[6]*a[11]+a[12]*a[7]*a[10];
    inv[8] = a[4]*a[9]*a[15]-a[4]*a[11]*a[13]-a[8]*a[5]*a[15]+a[8]*a[7]*a[13]+a[12]*a[5]*a[11]-a[12]*a[7]*a[9];
    inv[12] = -a[4]*a[9]*a[14]+a[4]*a[10]*a[13]+a[8]*a[5]*a[14]-a[8]*a[6]*a[13]-a[12]*a[5]*a[10]+a[12]*a[6]*a[9];
    inv[1] = -a[1]*a[10]*a[15]+a[1]*a[11]*a[14]+a[9]*a[2]*a[15]-a[9]*a[3]*a[14]-a[13]*a[2]*a[11]+a[13]*a[3]*a[10];
    inv[5] = a[0]*a[10]*a[15]-a[0]*a[11]*a[14]-a[8]*a[2]*a[15]+a[8]*a[3]*a[14]+a[12]*a[2]*a[11]-a[12]*a[3]*a[10];
    inv[9] = -a[0]*a[9]*a[15]+a[0]*a[11]*a[13]+a[8]*a[1]*a[15]-a[8]*a[3]*a[13]-a[12]*a[1]*a[11]+a[12]*a[3]*a[9];
    inv[13] = a[0]*a[9]*a[14]-a[0]*a[10]*a[13]-a[8]*a[1]*a[14]+a[8]*a[2]*a[13]+a[12]*a[1]*a[10]-a[12]*a[2]*a[9];
    inv[2] = a[1]*a[6]*a[15]-a[1]*a[7]*a[14]-a[5]*a[2]*a[15]+a[5]*a[3]*a[14]+a[13]*a[2]*a[7]-a[13]*a[3]*a[6];
    inv[6] = -a[0]*a[6]*a[15]+a[0]*a[7]*a[14]+a[4]*a[2]*a[15]-a[4]*a[3]*a[14]-a[12]*a[2]*a[7]+a[12]*a[3]*a[6];
    inv[10] = a[0]*a[5]*a[15]-a[0]*a[7]*a[13]-a[4]*a[1]*a[15]+a[4]*a[3]*a[13]+a[12]*a[1]*a[7]-a[12]*a[3]*a[5];
    inv[14] = -a[0]*a[5]*a[14]+a[0]*a[6]*a[13]+a[4]*a[1]*a[14]-a[4]*a[2]*a[13]-a[12]*a[1]*a[6]+a[12]*a[2]*a[5];
    inv[3] = -a[1]*a[6]*a[11]+a[1]*a[7]*a[10]+a[5]*a[2]*a[11]-a[5]*a[3]*a[10]-a[9]*a[2]*a[7]+a[9]*a[3]*a[6];
    inv[7] = a[0]*a[6]*a[11]-a[0]*a[7]*a[10]-a[4]*a[2]*a[11]+a[4]*a[3]*a[10]+a[8]*a[2]*a[7]-a[8]*a[3]*a[6];
    inv[11] = -a[0]*a[5]*a[11]+a[0]*a[7]*a[9]+a[4]*a[1]*a[11]-a[4]*a[3]*a[9]-a[8]*a[1]*a[7]+a[8]*a[3]*a[5];
    inv[15] = a[0]*a[5]*a[10]-a[0]*a[6]*a[9]-a[4]*a[1]*a[10]+a[4]*a[2]*a[9]+a[8]*a[1]*a[6]-a[8]*a[2]*a[5];
    let det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    if det.abs() < 1e-300 || !det.is_finite() { return None; }
    Some(std::array::from_fn(|r| std::array::from_fn(|c| inv[r * 4 + c] / det)))
}

/// Camera as stored in `metadata.json`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CameraMeta {
    /// Matrix used to rasterize the depth buffer.
    pub world_to_clip: Mat4,
    /// Derived from the matrix when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<Vec3>,
}

#[derive(Clone, Debug)]
pub struct Camera { pub world_to_clip: Mat4, pub clip_to_world: Mat4, pub position: Vec3, pub width: usize, pub height: usize }

impl Camera {
    pub fn new(meta: &CameraMeta, width: usize, height: usize) -> anyhow::Result<Self> {
        let m = meta.world_to_clip;
        let clip_to_world = invert(&m).ok_or_else(|| anyhow::anyhow!("world_to_clip is singular"))?;
        let position = match meta.position { Some(p) => p, None => centre(&m)? };
        Ok(Self { world_to_clip: m, clip_to_world, position, width, height })
    }
    /// World point to (pixel x, pixel y, ndc z). None behind the camera.
    pub fn project(&self, p: Vec3) -> Option<(f64, f64, f64)> {
        let c = mul4(&self.world_to_clip, [p[0], p[1], p[2], 1.0]);
        if c[3] <= 1e-12 { return None; }
        Some(((c[0] / c[3] * 0.5 + 0.5) * self.width as f64, (0.5 - c[1] / c[3] * 0.5) * self.height as f64, c[2] / c[3]))
    }
    /// Direction to pixel coordinates (point at infinity). None behind the camera.
    pub fn project_direction(&self, d: Vec3) -> Option<(f64, f64)> {
        let c = mul4(&self.world_to_clip, [d[0], d[1], d[2], 0.0]);
        if c[3] <= 1e-12 { return None; }
        Some(((c[0] / c[3] * 0.5 + 0.5) * self.width as f64, (0.5 - c[1] / c[3] * 0.5) * self.height as f64))
    }
    fn ndc_xy(&self, px: f64, py: f64) -> (f64, f64) { (px / self.width as f64 * 2.0 - 1.0, 1.0 - py / self.height as f64 * 2.0) }
    /// Pixel position and NDC depth to world. None at infinity.
    pub fn unproject(&self, px: f64, py: f64, ndc_z: f64) -> Option<Vec3> {
        let (x, y) = self.ndc_xy(px, py);
        let h = mul4(&self.clip_to_world, [x, y, ndc_z, 1.0]);
        if h[3].abs() < 1e-12 { return None; }
        let p = [h[0] / h[3], h[1] / h[3], h[2] / h[3]];
        p.iter().all(|v| v.is_finite()).then_some(p)
    }
    /// Unit ray direction through a pixel position.
    pub fn direction(&self, px: f64, py: f64) -> Vec3 {
        let (x, y) = self.ndc_xy(px, py);
        // Finite for standard, reversed and infinite projections.
        let a = mul4(&self.clip_to_world, [x, y, 0.3, 1.0]);
        let b = mul4(&self.clip_to_world, [x, y, 0.7, 1.0]);
        let pa = [a[0] / a[3], a[1] / a[3], a[2] / a[3]];
        let pb = [b[0] / b[3], b[1] / b[3], b[2] / b[3]];
        let d = normalize(sub(pb, pa));
        if dot(d, sub(pa, self.position)) < 0.0 { [-d[0], -d[1], -d[2]] } else { d }
    }
}

/// Camera centre: the point where clip x = y = w = 0.
fn centre(m: &Mat4) -> anyhow::Result<Vec3> {
    let rows = [m[0], m[1], m[3]];
    let a: [[f64; 3]; 3] = std::array::from_fn(|r| [rows[r][0], rows[r][1], rows[r][2]]);
    let b: Vec3 = std::array::from_fn(|r| -rows[r][3]);
    let det = |a: &[[f64; 3]; 3]| a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let d = det(&a);
    anyhow::ensure!(d.abs() > 1e-300, "cannot derive camera centre: provide camera.position");
    Ok(std::array::from_fn(|k| { let mut ak = a; for r in 0..3 { ak[r][k] = b[r]; } det(&ak) / d }))
}

/// Left-handed look-at with reversed-Z infinite perspective (Unreal-like).
pub fn look_at_reversed_infinite(eye: Vec3, target: Vec3, up: Vec3, fov_y_deg: f64, aspect: f64, near: f64) -> Mat4 {
    let f = normalize(sub(target, eye));
    let r = normalize(cross(up, f));
    let u = cross(f, r);
    let view: Mat4 = [[r[0], r[1], r[2], -dot(r, eye)], [u[0], u[1], u[2], -dot(u, eye)], [f[0], f[1], f[2], -dot(f, eye)], [0.0, 0.0, 0.0, 1.0]];
    let sy = 1.0 / (fov_y_deg.to_radians() * 0.5).tan();
    let proj: Mat4 = [[sy / aspect, 0.0, 0.0, 0.0], [0.0, sy, 0.0, 0.0], [0.0, 0.0, 0.0, near], [0.0, 0.0, 1.0, 0.0]];
    matmul(&proj, &view)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cam() -> Camera {
        let m = look_at_reversed_infinite([3.0, 2.0, -5.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0], 60.0, 16.0 / 9.0, 0.1);
        Camera::new(&CameraMeta { world_to_clip: m, position: None }, 320, 180).unwrap()
    }
    #[test]
    fn centre_is_derived_from_the_matrix() {
        let c = cam();
        for k in 0..3 { assert!((c.position[k] - [3.0, 2.0, -5.0][k]).abs() < 1e-9); }
    }
    #[test]
    fn project_unproject_roundtrip() {
        let c = cam();
        for p in [[0.0, 0.5, 0.0], [1.0, 0.0, 2.0], [-2.0, 1.5, 1.0]] {
            let (x, y, z) = c.project(p).unwrap();
            let q = c.unproject(x, y, z).unwrap();
            assert!(length(sub(p, q)) < 1e-9, "{p:?} -> {q:?}");
        }
        // Reversed infinite depth: ndc 0 is the point at infinity.
        assert!(c.unproject(160.0, 90.0, 0.0).is_none());
    }
    #[test]
    fn direction_points_through_pixel() {
        let c = cam();
        let (x, y, _) = c.project([0.0, 0.5, 0.0]).unwrap();
        let d = c.direction(x, y);
        let expected = normalize(sub([0.0, 0.5, 0.0], c.position));
        assert!(length(sub(d, expected)) < 1e-9);
        let (u, v) = c.project_direction(d).unwrap();
        assert!((u - x).abs() < 1e-6 && (v - y).abs() < 1e-6);
    }
}
