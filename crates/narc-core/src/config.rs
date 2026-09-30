use serde::{Deserialize,Serialize};
use crate::scene::Sequence;
#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(default,deny_unknown_fields)]
pub struct Config {pub render:Render,pub teacher:TeacherConfig,pub cache:CacheConfig,pub benchmark:Benchmark}
#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(default,deny_unknown_fields)]
pub struct Render {pub width:usize,pub height:usize,pub frames:usize,pub sequence:Sequence}
#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(default,deny_unknown_fields)]
pub struct TeacherConfig {pub seed:u64,pub dtype:String,pub hidden:Vec<usize>}
#[derive(Clone,Debug,Serialize,Deserialize,PartialEq,Eq)]
#[serde(default,deny_unknown_fields)]
pub struct CacheConfig {pub position_grid:[usize;3],pub view_bins:usize,pub light_bins:usize,pub dtype:String,pub interpolation:String}
#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(default,deny_unknown_fields)]
pub struct Benchmark {pub warmup_frames:usize,pub measured_frames:usize,pub cuda_graph:bool,pub max_vram_gib:f64}
impl Default for Render{fn default()->Self{Self{width:1920,height:1080,frames:500,sequence:Sequence::Camera}}}
impl Default for TeacherConfig{fn default()->Self{Self{seed:1337,dtype:"f32".into(),hidden:vec![128,128,128,64]}}}
impl Default for CacheConfig{fn default()->Self{Self{position_grid:[24;3],view_bins:24,light_bins:12,dtype:"f32".into(),interpolation:"nearest".into()}}}
impl Default for Benchmark{fn default()->Self{Self{warmup_frames:50,measured_frames:500,cuda_graph:true,max_vram_gib:10.0}}}
impl Default for Config{fn default()->Self{Self{render:Render::default(),teacher:TeacherConfig::default(),cache:CacheConfig::default(),benchmark:Benchmark::default()}}}
impl CacheConfig {
    pub fn entries(&self)->anyhow::Result<usize>{[6,self.position_grid[0],self.position_grid[1],self.position_grid[2],self.view_bins,self.light_bins].into_iter().try_fold(1usize,|a,b|{anyhow::ensure!(b>0,"cache dimensions must be positive");a.checked_mul(b).ok_or_else(||anyhow::anyhow!("cache size overflow"))})}
    pub fn bytes(&self)->anyhow::Result<usize>{self.entries()?.checked_mul(16).ok_or_else(||anyhow::anyhow!("cache bytes overflow"))}
}
impl Config {
    pub fn load(path:&std::path::Path)->anyhow::Result<Self>{Ok(toml::from_str(&std::fs::read_to_string(path)?)?)}
    pub fn working_set_bytes(&self)->anyhow::Result<usize>{let n=self.render.width.checked_mul(self.render.height).ok_or_else(||anyhow::anyhow!("resolution overflow"))?;let pixels=n.checked_mul(4*(32+128+128+64+32+4+1)+4).ok_or_else(||anyhow::anyhow!("working set overflow"))?;pixels.checked_add(self.cache.bytes()?).and_then(|v|v.checked_add(256*1024*1024)).ok_or_else(||anyhow::anyhow!("working set overflow"))}
    pub fn validate(&self)->anyhow::Result<()> {
        anyhow::ensure!(self.render.width>0&&self.render.height>0&&self.render.frames>0,"positive resolution and frames required");
        anyhow::ensure!(self.render.width<=16384&&self.render.height<=16384,"resolution too large");
        anyhow::ensure!((self.render.width*self.render.height)%32==0,"baseline requires a pixel count divisible by 32");
        anyhow::ensure!(self.teacher.hidden==[128,128,128,64]&&self.teacher.dtype=="f32","baseline requires fixed teacher and f32");
        anyhow::ensure!(self.cache.dtype=="f32"&&self.cache.interpolation=="nearest","baseline supports f32 nearest only");
        anyhow::ensure!(self.cache.entries()?<=i32::MAX as usize/4,"cache exceeds i32 gather addressing");
        anyhow::ensure!(self.benchmark.max_vram_gib.is_finite()&&self.benchmark.max_vram_gib>0.,"invalid VRAM budget");
        anyhow::ensure!(self.working_set_bytes()? as f64<=self.benchmark.max_vram_gib*1073741824.0,"estimated working set exceeds VRAM budget");Ok(())
    }
}
#[cfg(test)]mod tests{use super::*;#[test]fn vram_estimate(){let mut c=Config::default();assert_eq!(c.cache.bytes().unwrap(),6*24*24*24*24*12*16);assert!(c.validate().is_ok());c.cache.position_grid=[usize::MAX;3];assert!(c.validate().is_err());}#[test]fn zero_dimension(){let mut c=Config::default();c.cache.view_bins=0;assert!(c.validate().is_err());}}
