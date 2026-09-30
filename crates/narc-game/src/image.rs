//! RGB float images, PNG and OpenEXR I/O.
use std::path::Path;

/// Interleaved RGB f32. Color space is tracked by the caller.
#[derive(Clone, Debug, PartialEq)]
pub struct Rgb { pub width: usize, pub height: usize, pub data: Vec<f32> }

impl Rgb {
    pub fn new(width: usize, height: usize) -> Self { Self { width, height, data: vec![0.0; width * height * 3] } }
    pub fn px(&self, x: usize, y: usize) -> [f32; 3] { let i = (y * self.width + x) * 3; [self.data[i], self.data[i + 1], self.data[i + 2]] }
    pub fn set(&mut self, x: usize, y: usize, c: [f32; 3]) { let i = (y * self.width + x) * 3; self.data[i..i + 3].copy_from_slice(&c); }
    /// Clamped bilinear sample, pixel centres at +0.5.
    pub fn bilinear(&self, fx: f64, fy: f64) -> [f32; 3] {
        let x = (fx - 0.5).clamp(0.0, (self.width - 1) as f64);
        let y = (fy - 0.5).clamp(0.0, (self.height - 1) as f64);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (wx, wy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
        let (a, b, c, d) = (self.px(x0, y0), self.px(x1, y0), self.px(x0, y1), self.px(x1, y1));
        std::array::from_fn(|k| (a[k] * (1.0 - wx) + b[k] * wx) * (1.0 - wy) + (c[k] * (1.0 - wx) + d[k] * wx) * wy)
    }
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Self { Self { width: self.width, height: self.height, data: self.data.iter().map(|&v| f(v)).collect() } }
    pub fn to_linear(&self) -> Self { self.map(srgb_to_linear) }
    pub fn to_srgb(&self) -> Self { self.map(linear_to_srgb) }
}

pub fn srgb_to_linear(c: f32) -> f32 { let c = c.clamp(0.0, 1.0); if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) } }
pub fn linear_to_srgb(c: f32) -> f32 { let c = c.clamp(0.0, 1.0); if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 } }

/// PNG to sRGB floats in [0, 1].
pub fn load_color(path: &Path) -> anyhow::Result<Rgb> {
    let img = image::open(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?.to_rgb32f();
    Ok(Rgb { width: img.width() as usize, height: img.height() as usize, data: img.into_raw() })
}
pub fn save_png(path: &Path, img: &Rgb) -> anyhow::Result<()> {
    let bytes: Vec<u8> = img.data.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    image::save_buffer(path, &bytes, img.width as u32, img.height as u32, image::ColorType::Rgb8)?;
    Ok(())
}
fn load_exr_rgba(path: &Path) -> anyhow::Result<(usize, usize, Vec<f32>)> {
    let img = image::open(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?.to_rgba32f();
    Ok((img.width() as usize, img.height() as usize, img.into_raw()))
}
/// Channel R of a float EXR.
pub fn load_scalar_exr(path: &Path) -> anyhow::Result<(usize, usize, Vec<f32>)> {
    let (w, h, d) = load_exr_rgba(path)?;
    Ok((w, h, d.chunks_exact(4).map(|p| p[0]).collect()))
}
pub fn load_vec_exr<const N: usize>(path: &Path) -> anyhow::Result<(usize, usize, Vec<[f32; N]>)> {
    let (w, h, d) = load_exr_rgba(path)?;
    Ok((w, h, d.chunks_exact(4).map(|p| std::array::from_fn(|k| p[k])).collect()))
}
/// Up to three float channels to an RGB EXR.
pub fn save_exr(path: &Path, width: usize, height: usize, channels: usize, data: &[f32]) -> anyhow::Result<()> {
    anyhow::ensure!(data.len() == width * height * channels && (1..=3).contains(&channels), "EXR data size mismatch");
    let rgb: Vec<f32> = data.chunks_exact(channels).flat_map(|p| std::array::from_fn::<f32, 3, _>(|k| if k < channels { p[k] } else if channels == 1 { p[0] } else { 0.0 })).collect();
    let buf = image::Rgb32FImage::from_raw(width as u32, height as u32, rgb).ok_or_else(|| anyhow::anyhow!("EXR buffer"))?;
    image::DynamicImage::ImageRgb32F(buf).save(path)?;
    Ok(())
}

/// Max abs RGB error per pixel, white at `scale`.
pub fn heatmap(a: &Rgb, b: &Rgb, scale: f32) -> Rgb {
    let mut out = Rgb::new(a.width, a.height);
    for (i, (x, y)) in a.data.chunks_exact(3).zip(b.data.chunks_exact(3)).enumerate() {
        let e = x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
        let t = (e / scale).clamp(0.0, 1.0) * 3.0;
        out.data[i * 3..i * 3 + 3].copy_from_slice(&[t.min(1.0), (t - 1.0).clamp(0.0, 1.0), (t - 2.0).clamp(0.0, 1.0)]);
    }
    out
}
pub fn mask(width: usize, height: usize, on: impl Fn(usize) -> bool) -> Rgb {
    let mut m = Rgb::new(width, height);
    for i in 0..width * height { if on(i) { m.data[i * 3..i * 3 + 3].copy_from_slice(&[1.0; 3]); } }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn srgb_roundtrip() { for v in [0.0, 0.001, 0.2, 0.5, 1.0] { assert!((linear_to_srgb(srgb_to_linear(v)) - v).abs() < 1e-5); } }
    #[test]
    fn exr_png_roundtrip() {
        let dir = std::env::temp_dir().join(format!("narc-game-img-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let depth: Vec<f32> = (0..12).map(|i| i as f32 * 0.125).collect();
        save_exr(&dir.join("d.exr"), 4, 3, 1, &depth).unwrap();
        assert_eq!(load_scalar_exr(&dir.join("d.exr")).unwrap().2, depth);
        let mut c = Rgb::new(2, 2); c.set(1, 1, [1.0, 0.5, 0.0]);
        save_png(&dir.join("c.png"), &c).unwrap();
        let back = load_color(&dir.join("c.png")).unwrap();
        assert!((back.px(1, 1)[1] - 128.0 / 255.0).abs() < 1e-6);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn bilinear_at_centre_is_exact() {
        let mut c = Rgb::new(3, 3); c.set(1, 1, [0.9, 0.1, 0.4]);
        assert_eq!(c.bilinear(1.5, 1.5), [0.9, 0.1, 0.4]);
    }
}
