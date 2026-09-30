use narc_core::{scene::Sequence,teacher::Teacher,config::CacheConfig,metrics::ErrorMetrics};
use cutile::prelude::*;
fn main()->anyhow::Result<()> {
    let teacher=Teacher::new(1337);let c=CacheConfig{position_grid:[4;3],view_bins:8,light_bins:4,..Default::default()};
    let mut gpu=narc_gpu::Gpu::new(96,32,&teacher)?;
    gpu.bake(&c,&teacher)?;
    let cache=api::dup(gpu.cache.as_ref().unwrap()).to_host_vec().sync_on(&gpu.stream)?;
    for i in (0..c.entries()?).step_by(97){let cpu=teacher.forward(&narc_core::cache::representative(&c,i));let e=ErrorMetrics::compare(&cpu,&cache[i*4..i*4+3]);anyhow::ensure!(e.max_abs_error<0.0002,"bake differs at {i}: {e:?}");}
    for seq in [Sequence::Static,Sequence::Camera,Sequence::CameraLight]{
        gpu.reset(120)?;gpu.full(seq)?;let full=gpu.read_rgb()?;
        gpu.reset(120)?;gpu.cached(seq,&c)?;let cached=gpu.read_cache_rgb()?;
        let keys=api::dup(&gpu.keys).to_host_vec().sync_on(&gpu.stream)?;
        for (i,&key) in keys.iter().enumerate(){let(f,o)=narc_core::scene::sample(96,32,i,120,seq);anyhow::ensure!(Some(key as usize)==narc_core::cache::quantize(&c,o as usize,&f),"key mismatch at {i}");}
        let e=ErrorMetrics::compare(&full,&cached);let (s,m)=gpu.error_sums()?;
        anyhow::ensure!((s/(96.*32.*3.)-e.mse).abs()<1e-7&&(m-e.max_abs_error).abs()<1e-7,"GPU reduction differs");println!("{seq:?} nearest vs FULL {e:?}");
        gpu.reset(120)?;gpu.cached_interp(seq,&c)?;let interp=gpu.read_cache_rgb()?;
        let cpu:Vec<f32>=(0..96*32).flat_map(|i|{let(f,o)=narc_core::scene::sample(96,32,i,120,seq);narc_core::cache::interpolate(&c,o as usize,&f,&cache)}).collect();
        let r=ErrorMetrics::compare(&interp,&cpu);anyhow::ensure!(r.max_abs_error<2e-4,"interpolated lookup differs from CPU reference: {r:?}");
        println!("{seq:?} interp GPU vs CPU max {:.2e}; interp vs FULL {:?}",r.max_abs_error,ErrorMetrics::compare(&full,&interp));
    }println!("GPU bake, quantization, nearest and multilinear lookup, error reduction passed");Ok(())
}
