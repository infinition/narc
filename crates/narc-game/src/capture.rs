//! Capture format and the FrameSource trait.
//!
//! ```text
//! <capture>/
//!   capture.json            manifest
//!   frame_000000/
//!     color.png             required, final LDR color, sRGB
//!     depth.exr             level 1+, raw NDC depth in channel R (as in the depth buffer)
//!     normal.exr            level 2+, world-space normal in RGB
//!     motion.exr            level 3+, screen motion in pixels in RG
//!     gbuffer/              level 4, free-form extra buffers (base color, roughness, ...)
//!     metadata.json         index, time, camera (world_to_clip), tags
//! ```
use crate::camera::{Camera, CameraMeta};
use crate::image::{self, Rgb};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const FORMAT: &str = "narc-capture";
pub const VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DepthSpec {
    /// Only "ndc" (raw depth buffer value).
    pub kind: String,
    /// Cleared depth value: 0 for reversed Z, 1 for standard Z.
    pub sky_ndc: Option<f32>,
}

/// Set by whoever records the capture.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum SceneClass { StaticCacheable, DynamicUnsafe, Unknown }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub format: String,
    pub version: u32,
    pub name: String,
    /// "mock", "renderdoc", ...
    pub source: String,
    pub width: usize,
    pub height: usize,
    pub depth: DepthSpec,
    pub scene_class: SceneClass,
    /// Frame directories in temporal order.
    pub frames: Vec<String>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FrameMeta {
    pub index: usize,
    pub time_s: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<CameraMeta>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Free-form data kept for audit.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub extra: serde_json::Value,
}

/// Cumulative data levels.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum DataLevel {
    /// color only
    L0Color,
    /// color + depth
    L1Depth,
    /// + normals
    L2Normals,
    /// + motion vectors
    L3Motion,
    /// + G-buffer / material / light metadata
    L4GBuffer,
}
impl DataLevel {
    pub fn describe(self) -> &'static str {
        match self { Self::L0Color => "LEVEL 0 color", Self::L1Depth => "LEVEL 1 color+depth", Self::L2Normals => "LEVEL 2 +normals", Self::L3Motion => "LEVEL 3 +motion vectors", Self::L4GBuffer => "LEVEL 4 +gbuffer/metadata" }
    }
}

/// One captured sequence. Missing optional buffers return Ok(None).
pub trait FrameSource {
    fn manifest(&self) -> &Manifest;
    fn len(&self) -> usize { self.manifest().frames.len() }
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn size(&self) -> (usize, usize) { (self.manifest().width, self.manifest().height) }
    /// Highest level present on every frame.
    fn level(&self) -> anyhow::Result<DataLevel>;
    /// sRGB-encoded color in [0, 1].
    fn color(&self, i: usize) -> anyhow::Result<Rgb>;
    /// Raw NDC depth, row-major.
    fn depth(&self, i: usize) -> anyhow::Result<Option<Vec<f32>>>;
    fn normals(&self, i: usize) -> anyhow::Result<Option<Vec<[f32; 3]>>>;
    fn motion_vectors(&self, i: usize) -> anyhow::Result<Option<Vec<[f32; 2]>>>;
    fn camera(&self, i: usize) -> anyhow::Result<Option<Camera>>;
    fn metadata(&self, i: usize) -> anyhow::Result<FrameMeta>;
}

/// A capture directory on disk.
pub struct DiskCapture { pub root: PathBuf, pub manifest: Manifest }

impl DiskCapture {
    pub fn open(root: &Path) -> anyhow::Result<Self> {
        let path = root.join("capture.json");
        let manifest: Manifest = serde_json::from_slice(&std::fs::read(&path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?)?;
        anyhow::ensure!(manifest.format == FORMAT, "{}: format must be \"{FORMAT}\"", path.display());
        anyhow::ensure!(manifest.version == VERSION, "{}: unsupported version {}", path.display(), manifest.version);
        anyhow::ensure!(manifest.depth.kind == "ndc", "depth.kind must be \"ndc\"");
        anyhow::ensure!(manifest.width > 0 && manifest.height > 0 && !manifest.frames.is_empty(), "empty capture");
        Ok(Self { root: root.to_path_buf(), manifest })
    }
    pub fn frame_dir(&self, i: usize) -> PathBuf { self.root.join(&self.manifest.frames[i]) }
    fn file(&self, i: usize, name: &str) -> Option<PathBuf> { let p = self.frame_dir(i).join(name); p.exists().then_some(p) }
    fn check(&self, w: usize, h: usize, what: &str, i: usize) -> anyhow::Result<()> {
        anyhow::ensure!((w, h) == (self.manifest.width, self.manifest.height), "frame {i} {what}: {w}x{h}, manifest says {}x{}", self.manifest.width, self.manifest.height);
        Ok(())
    }
    /// Level of a single frame, from the files present.
    pub fn frame_level(&self, i: usize) -> anyhow::Result<DataLevel> {
        anyhow::ensure!(self.file(i, "color.png").is_some(), "frame {i}: color.png missing");
        let has = |n: &str| self.file(i, n).is_some();
        let meta = self.metadata(i)?;
        Ok(if !has("depth.exr") { DataLevel::L0Color }
            else if !has("normal.exr") { DataLevel::L1Depth }
            else if !has("motion.exr") { DataLevel::L2Normals }
            else if !(self.frame_dir(i).join("gbuffer").is_dir() || !meta.extra.is_null()) { DataLevel::L3Motion }
            else { DataLevel::L4GBuffer })
    }
}

impl FrameSource for DiskCapture {
    fn manifest(&self) -> &Manifest { &self.manifest }
    fn level(&self) -> anyhow::Result<DataLevel> {
        (0..self.len()).map(|i| self.frame_level(i)).try_fold(DataLevel::L4GBuffer, |a, b| Ok(a.min(b?)))
    }
    fn color(&self, i: usize) -> anyhow::Result<Rgb> {
        let c = image::load_color(&self.frame_dir(i).join("color.png"))?;
        self.check(c.width, c.height, "color", i)?;
        Ok(c)
    }
    fn depth(&self, i: usize) -> anyhow::Result<Option<Vec<f32>>> {
        let Some(p) = self.file(i, "depth.exr") else { return Ok(None) };
        let (w, h, d) = image::load_scalar_exr(&p)?;
        self.check(w, h, "depth", i)?;
        Ok(Some(d))
    }
    fn normals(&self, i: usize) -> anyhow::Result<Option<Vec<[f32; 3]>>> {
        let Some(p) = self.file(i, "normal.exr") else { return Ok(None) };
        let (w, h, d) = image::load_vec_exr::<3>(&p)?;
        self.check(w, h, "normal", i)?;
        Ok(Some(d))
    }
    fn motion_vectors(&self, i: usize) -> anyhow::Result<Option<Vec<[f32; 2]>>> {
        let Some(p) = self.file(i, "motion.exr") else { return Ok(None) };
        let (w, h, d) = image::load_vec_exr::<2>(&p)?;
        self.check(w, h, "motion", i)?;
        Ok(Some(d))
    }
    fn camera(&self, i: usize) -> anyhow::Result<Option<Camera>> {
        self.metadata(i)?.camera.map(|m| Camera::new(&m, self.manifest.width, self.manifest.height)).transpose()
    }
    fn metadata(&self, i: usize) -> anyhow::Result<FrameMeta> {
        let p = self.frame_dir(i).join("metadata.json");
        Ok(serde_json::from_slice(&std::fs::read(&p).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?)?)
    }
}

pub fn write_frame(root: &Path, dir: &str, color: &Rgb, depth: Option<&[f32]>, normals: Option<&[f32]>, meta: &FrameMeta) -> anyhow::Result<()> {
    let d = root.join(dir);
    std::fs::create_dir_all(&d)?;
    image::save_png(&d.join("color.png"), color)?;
    if let Some(z) = depth { image::save_exr(&d.join("depth.exr"), color.width, color.height, 1, z)?; }
    if let Some(n) = normals { image::save_exr(&d.join("normal.exr"), color.width, color.height, 3, n)?; }
    std::fs::write(d.join("metadata.json"), serde_json::to_vec_pretty(meta)?)?;
    Ok(())
}
pub fn frame_dir_name(i: usize) -> String { format!("frame_{i:06}") }
