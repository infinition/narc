use narc_core::{scene::Sequence,teacher::Teacher,config::CacheConfig};
fn main()->anyhow::Result<()> {
    let teacher=Teacher::new(1337);let c=CacheConfig{position_grid:[4;3],view_bins:8,light_bins:4,..Default::default()};let mut gpu=narc_gpu::Gpu::new(96,32,&teacher)?;gpu.bake(&c,&teacher)?;
    for seq in [Sequence::Static,Sequence::Camera,Sequence::CameraLight] {
        gpu.capture(seq,&c)?;gpu.reset(119)?;gpu.replay()?;gpu.replay()?;let graph=gpu.read_cache_rgb()?;
        gpu.reset(120)?;gpu.cached(seq,&c)?;let normal=gpu.read_cache_rgb()?;anyhow::ensure!(graph==normal,"graph frame state differs");
    }
    gpu.invalidate_object(2)?;gpu.reset(120)?;gpu.cached(Sequence::Camera,&c)?;let cached=gpu.read_cache_rgb()?;gpu.reset(120)?;gpu.full(Sequence::Camera)?;let full=gpu.read_rgb()?;
    for i in 0..96*32{let (_,o)=narc_core::scene::sample(96,32,i,120,Sequence::Camera);if o==2{anyhow::ensure!(cached[3*i..3*i+3]==full[3*i..3*i+3],"invalidated object did not fall back");}}
    anyhow::ensure!(gpu.hit_count()?==96*32*5/6,"hit mask wrong after invalidation");
    std::fs::create_dir_all("out/validation")?;gpu.save_cache(std::path::Path::new("out/validation/roundtrip.narc"),&c,&teacher)?;gpu.load_cache(std::path::Path::new("out/validation/roundtrip.narc"),&c,&teacher)?;anyhow::ensure!(gpu.invalid_mask==4,"persisted invalidation lost");
    println!("Graph motion, eager equality, object fallback, GPU hit mask and disk roundtrip passed");Ok(())
}
