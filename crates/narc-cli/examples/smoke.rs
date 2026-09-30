use cutile::prelude::*;

#[cutile::module]
mod smoke {
    use cutile::core::*;
    #[cutile::entry()]
    fn add<const B: i32>(z: &mut Tensor<f32, {[B]}>, x: &Tensor<f32, {[-1]}>, y: &Tensor<f32, {[-1]}>) {
        z.store(x.load_like(z) + y.load_like(z));
    }
}

fn main() -> anyhow::Result<()> {
    let device = cuda_core::Device::new(0)?;
    println!("GPU: {}", device.name()?);
    let x = api::ones::<f32>(&[1024]);
    let y = api::ones::<f32>(&[1024]);
    let z = api::zeros::<f32>(&[1024]).partition([128]);
    let (z, _, _) = smoke::add(z, x, y).sync()?;
    let values = z.unpartition().to_host_vec().sync()?;
    anyhow::ensure!(values.iter().all(|&v| v == 2.0), "cuTile smoke test failed");
    println!("cuTile 0.3.1 official README add example: 1024/1024 correct");
    Ok(())
}
