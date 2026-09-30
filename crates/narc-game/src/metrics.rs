//! Reconstruction metrics on sRGB images in [0, 1].
use crate::image::Rgb;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Quality {
    /// All pixels, holes filled.
    pub mse: f64,
    pub rmse: f64,
    pub psnr_db: Option<f64>,
    pub max_abs_error: f64,
    /// Luma, 11x11 Gaussian window, sigma 1.5.
    pub ssim: f64,
    /// Fraction of reconstructed pixels.
    pub coverage: f64,
    pub psnr_covered_db: Option<f64>,
}

pub fn psnr(mse: f64) -> Option<f64> { (mse > 0.0).then(|| -10.0 * mse.log10()) }

pub fn compare(reference: &Rgb, recon: &Rgb, covered: &[bool]) -> Quality {
    assert_eq!(reference.data.len(), recon.data.len());
    let (mut sum, mut max, mut sum_c, mut n_c) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    for (i, (a, b)) in reference.data.chunks_exact(3).zip(recon.data.chunks_exact(3)).enumerate() {
        let mut s = 0.0;
        for k in 0..3 { let d = (a[k] - b[k]) as f64; s += d * d; max = max.max(d.abs()); }
        sum += s;
        if covered[i] { sum_c += s; n_c += 1; }
    }
    let n = reference.width * reference.height;
    let mse = sum / (n * 3) as f64;
    Quality { mse, rmse: mse.sqrt(), psnr_db: psnr(mse), max_abs_error: max, ssim: ssim(reference, recon), coverage: n_c as f64 / n as f64, psnr_covered_db: if n_c == 0 { None } else { psnr(sum_c / (n_c * 3) as f64) } }
}

fn luma(img: &Rgb) -> Vec<f64> { img.data.chunks_exact(3).map(|p| 0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64).collect() }
fn blur(v: &[f64], w: usize, h: usize) -> Vec<f64> {
    let k: Vec<f64> = (-5..=5).map(|i: i32| (-(i * i) as f64 / (2.0 * 1.5 * 1.5)).exp()).collect();
    let s: f64 = k.iter().sum();
    let k: Vec<f64> = k.iter().map(|x| x / s).collect();
    let mut tmp = vec![0.0; w * h];
    for y in 0..h { for x in 0..w { tmp[y * w + x] = (0..11).map(|j| k[j] * v[y * w + (x as i64 + j as i64 - 5).clamp(0, w as i64 - 1) as usize]).sum(); } }
    let mut out = vec![0.0; w * h];
    for y in 0..h { for x in 0..w { out[y * w + x] = (0..11).map(|j| k[j] * tmp[(y as i64 + j as i64 - 5).clamp(0, h as i64 - 1) as usize * w + x]).sum(); } }
    out
}
/// Mean SSIM on luma (Wang et al. 2004 constants).
pub fn ssim(a: &Rgb, b: &Rgb) -> f64 {
    let (w, h) = (a.width, a.height);
    let (x, y) = (luma(a), luma(b));
    let prod = |p: &[f64], q: &[f64]| p.iter().zip(q).map(|(u, v)| u * v).collect::<Vec<f64>>();
    let (mx, my) = (blur(&x, w, h), blur(&y, w, h));
    let (sxx, syy, sxy) = (blur(&prod(&x, &x), w, h), blur(&prod(&y, &y), w, h), blur(&prod(&x, &y), w, h));
    let (c1, c2) = (0.01f64.powi(2), 0.03f64.powi(2));
    (0..w * h).map(|i| {
        let (vx, vy, cxy) = (sxx[i] - mx[i] * mx[i], syy[i] - my[i] * my[i], sxy[i] - mx[i] * my[i]);
        ((2.0 * mx[i] * my[i] + c1) * (2.0 * cxy + c2)) / ((mx[i] * mx[i] + my[i] * my[i] + c1) * (vx + vy + c2))
    }).sum::<f64>() / (w * h) as f64
}
/// RMSE of (R_t - R_t-1) - (O_t - O_t-1): flicker that per-frame PSNR misses.
pub fn temporal_rmse(orig_prev: &Rgb, orig: &Rgb, rec_prev: &Rgb, rec: &Rgb) -> f64 {
    let n = orig.data.len();
    ((0..n).map(|i| { let d = (rec.data[i] - rec_prev.data[i]) as f64 - (orig.data[i] - orig_prev.data[i]) as f64; d * d }).sum::<f64>() / n as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn img(f: impl Fn(usize) -> f32) -> Rgb { let mut r = Rgb::new(16, 12); for (i, v) in r.data.iter_mut().enumerate() { *v = f(i); } r }
    #[test]
    fn identical_images() {
        let a = img(|i| (i % 7) as f32 / 7.0);
        let q = compare(&a, &a, &vec![true; 16 * 12]);
        assert_eq!(q.psnr_db, None);
        assert!((q.ssim - 1.0).abs() < 1e-9 && q.coverage == 1.0);
        assert_eq!(temporal_rmse(&a, &a, &a, &a), 0.0);
    }
    #[test]
    fn known_error() {
        let a = img(|_| 0.5);
        let b = img(|_| 0.6);
        let q = compare(&a, &b, &vec![false; 16 * 12]);
        assert!((q.rmse - 0.1).abs() < 1e-6 && (q.psnr_db.unwrap() - 20.0).abs() < 1e-4);
        assert_eq!(q.psnr_covered_db, None);
        assert!(ssim(&a, &img(|i| (i % 5) as f32 / 5.0)) < 0.9);
    }
}
