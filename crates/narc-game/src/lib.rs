//! Offline reconstruction of captured game frames (CPU reference):
//! capture import, depth reprojection baseline, NARC cache, metrics.
pub mod camera;
pub mod capture;
pub mod image;
pub mod metrics;
pub mod mock;
pub mod narc;
pub mod pipeline;
pub mod reproject;

pub use capture::{DataLevel, DiskCapture, FrameSource};

/// Per-pixel outcome of a reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Counted against coverage, filled afterwards.
    Hole,
    Surface,
    Sky,
    /// View bin missing, voxel mean used.
    ViewFallback,
}
impl Status { pub fn covered(self) -> bool { self != Status::Hole } }

/// Cleared or non-finite depth.
pub fn is_sky(z: f32, sky_ndc: Option<f32>) -> bool { !z.is_finite() || sky_ndc.is_some_and(|s| (z - s).abs() <= 1e-7) }

/// Runs `row(y)` on all cores, rows kept in order.
pub fn par_rows<T: Send>(height: usize, row: impl Fn(usize) -> Vec<T> + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(height.max(1));
    let chunk = height.div_ceil(threads);
    let parts: Vec<Vec<T>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads).map(|t| {
            let row = &row;
            s.spawn(move || (t * chunk..((t + 1) * chunk).min(height)).flat_map(row).collect::<Vec<T>>())
        }).collect();
        handles.into_iter().map(|h| h.join().expect("worker panicked")).collect()
    });
    parts.into_iter().flatten().collect()
}

/// Fills each hole with the nearest covered pixel on its row (left wins ties).
pub fn fill_holes(img: &mut image::Rgb, status: &[Status]) {
    let w = img.width;
    for y in 0..img.height {
        let row: Vec<usize> = (0..w).filter(|&x| status[y * w + x].covered()).collect();
        if row.is_empty() { continue; }
        let mut j = 0;
        for x in 0..w {
            if status[y * w + x].covered() { continue; }
            while j + 1 < row.len() && row[j + 1] <= x { j += 1; }
            let left = row[j];
            let src = if left > x { left } else if j + 1 < row.len() && row[j + 1] - x < x - left { row[j + 1] } else { left };
            let c = img.px(src, y);
            img.set(x, y, c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn par_rows_keeps_order() { assert_eq!(par_rows(37, |y| vec![y, y]), (0..37).flat_map(|y| [y, y]).collect::<Vec<_>>()); }
    #[test]
    fn hole_fill_uses_nearest_on_row() {
        let mut img = image::Rgb::new(5, 1);
        img.set(1, 0, [1.0; 3]); img.set(4, 0, [0.5; 3]);
        let s = [Status::Hole, Status::Surface, Status::Hole, Status::Hole, Status::Surface];
        fill_holes(&mut img, &s);
        let r: Vec<f32> = (0..5).map(|x| img.px(x, 0)[0]).collect();
        assert_eq!(r, vec![1.0, 1.0, 1.0, 0.5, 0.5]);
    }
}
