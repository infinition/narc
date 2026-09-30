use narc_core::{scene::Sequence,teacher::Teacher,metrics::quantile};
fn main()->anyhow::Result<()> {
    let mut gpu=narc_gpu::Gpu::new(1280,720,&Teacher::new(1337))?;
    for _ in 0..10{gpu.full(Sequence::Camera)?;}gpu.finish()?;
    let start=gpu.device.new_event()?;let end=gpu.device.new_event()?;let mut times=Vec::new();
    for _ in 0..30{start.record(&gpu.stream)?;gpu.full(Sequence::Camera)?;end.record(&gpu.stream)?;end.synchronize()?;times.push(start.elapsed_time(&end)? as f64);}
    println!("FULL 1280x720 median={} ms p95={} ms",quantile(&times,0.5),quantile(&times,0.95));
    let rgb=gpu.read_rgb()?;
    let data:Vec<u8>=rgb.iter().map(|v|(v.clamp(0.0,1.0)*255.0).round() as u8).collect();
    std::fs::create_dir_all("out/full-baseline")?;
    image::save_buffer("out/full-baseline/full.png",&data,1280,720,image::ColorType::Rgb8)?;
    std::fs::write("out/full-baseline/timings.json",serde_json::to_string_pretty(&times)?)?;
    Ok(())
}
