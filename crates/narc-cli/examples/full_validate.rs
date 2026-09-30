use narc_core::{scene::{sample,Sequence},teacher::Teacher,metrics::ErrorMetrics};
fn main()->anyhow::Result<()> {
    let teacher=Teacher::new(1337);
    let mut gpu=narc_gpu::Gpu::new(96,32,&teacher)?;
    for sequence in [Sequence::Static,Sequence::Camera,Sequence::CameraLight] {
        for frame in [0,120] {
            gpu.reset(frame)?;gpu.full(sequence)?;let a=gpu.read_rgb()?;
            gpu.reset(frame)?;gpu.full(sequence)?;let b=gpu.read_rgb()?;
            anyhow::ensure!(a==b,"FULL repeat differs");
            let cpu:Vec<f32>=(0..96*32).flat_map(|i|teacher.forward(&sample(96,32,i,frame as u32,sequence).0)).collect();
            let e=ErrorMetrics::compare(&a,&cpu);
            println!("{sequence:?} frame {frame}: {e:?}");
            anyhow::ensure!(e.max_abs_error<0.0002,"GPU vs independent FP32 CPU reference differs");
        }
    }
    println!("FULL determinism and CPU reference passed");Ok(())
}
