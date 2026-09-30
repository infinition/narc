pub mod kernels;
use cutile::prelude::*;
use cuda_core::{Device,Stream};
use std::sync::Arc;
use narc_core::{scene::Sequence,teacher::{Teacher,PADDED}};
use narc_core::config::CacheConfig;
use kernels::kernels as k;
fn layer_io<'a>(features:&'a Tensor<f32>,buffers:&'a mut[Tensor<f32>],l:usize)->(&'a Tensor<f32>,&'a mut Tensor<f32>) {
    if l==0{return(features,&mut buffers[0]);}
    let (i,o)=match l {1=>(0,1),2=>(1,0),3=>(0,2),4=>(2,3),_=>unreachable!()};
    if i<o {let(a,b)=buffers.split_at_mut(o);(&a[i],&mut b[0])} else {let(a,b)=buffers.split_at_mut(i);(&b[0],&mut a[o])}
}
/// Teacher output tile, overridable with NARC_TEACHER_TILE=BMxBN.
/// Default from the sweep in results/teacher-tiles.log.
pub fn teacher_tile()->(usize,usize) {
    static T:std::sync::OnceLock<(usize,usize)>=std::sync::OnceLock::new();
    *T.get_or_init(||std::env::var("NARC_TEACHER_TILE").ok().and_then(|s|{let(a,b)=s.split_once('x')?;Some((a.parse().ok()?,b.parse().ok()?))}).unwrap_or(DEFAULT_TEACHER_TILE))
}
pub const DEFAULT_TEACHER_TILE:(usize,usize)=(8,128);
fn layer_tile(l:usize)->[usize;2]{let(bm,bn)=teacher_tile();[bm,bn.min(PADDED[l+1])]}
fn layer_generics(l:usize)->Vec<String>{let t=layer_tile(l);vec![PADDED[l].to_string(),if l==4{"1"}else{"0"}.into(),t[0].to_string(),t[1].to_string()]}

pub struct Gpu {
    pub graph:Option<CudaGraph<()>>,
    pub full_graph:Option<CudaGraph<()>>,
    pub device:Arc<Device>,
    pub stream:Arc<Stream>,
    pub frame:Tensor<i32>,
    pub features:Tensor<f32>,
    pub weights:Vec<Tensor<f32>>,
    pub biases:Vec<Tensor<f32>>,
    pub layers:Vec<Tensor<f32>>,
    pub cache:Option<Tensor<f32>>,
    pub invalid_mask:i32,
    pub keys:Tensor<i32>,pub rgb_cache:Tensor<f32>,pub errors:Tensor<f32>,pub reduced:Tensor<f32>,
    pub width:usize,pub height:usize,pub n:usize,pub cache_bytes:usize,
}
pub fn upload(v:Vec<f32>,shape:&[usize],s:&Arc<Stream>)->anyhow::Result<Tensor<f32>> {
    Ok(api::copy_host_vec_to_device(&Arc::new(v)).sync_on(s)?.reshape(shape)?)
}
impl Gpu {
    pub fn new(width:usize,height:usize,teacher:&Teacher)->anyhow::Result<Self> {
        let device=Device::new(0)?;let stream=device.new_stream()?;
        let n=(width*height).div_ceil(32)*32;
        let frame=api::zeros(&[1]).sync_on(&stream)?;
        let features=api::zeros(&[n,32]).sync_on(&stream)?;
        let mut weights=Vec::new();let mut biases=Vec::new();let mut layers=Vec::new();
        for l in 0..5 {
            weights.push(upload(teacher.weights[l].clone(),&[PADDED[l],PADDED[l+1]],&stream)?);
            biases.push(upload(teacher.biases[l].clone(),&[PADDED[l+1]],&stream)?);
            
        }
        for d in [128,128,64,32] {layers.push(api::zeros(&[n,d]).sync_on(&stream)?);}
        let keys=api::zeros(&[n]).sync_on(&stream)?;
        let rgb_cache=api::zeros(&[n,4]).sync_on(&stream)?;
        let errors=api::zeros(&[n/32,2]).sync_on(&stream)?;
        let reduced=api::zeros(&[1,2]).sync_on(&stream)?;
        Ok(Self{graph:None,full_graph:None,device,stream,frame,features,weights,biases,layers,cache:None,invalid_mask:0,keys,rgb_cache,errors,reduced,width,height,n,cache_bytes:0})
    }
    pub fn reset(&mut self,frame:i32)->anyhow::Result<()> {
        // Scalar kernel, no copy.
        unsafe{k::set_frame((&mut self.frame).partition([1]),frame).async_on(&self.stream)?;}
        Ok(())
    }
    /// cuTile specializes on integer scalar divisibility (up to 16):
    /// compile every frame value class and sequence code up front.
    pub fn compile_runtime(&mut self,_sequence:Sequence,c:&CacheConfig)->anyhow::Result<()> {
        for v in [0,1,2,4,8] {k::set_frame((&mut self.frame).partition([1]),v).compile()?;}
        k::advance((&mut self.frame).partition([1])).compile()?;
        for s in [Sequence::Static,Sequence::Camera,Sequence::CameraLight] {k::generate_frame_features((&mut self.features).partition([32,32]),&self.frame,self.width as i32,self.height as i32,s.code()).compile()?;}
        for l in 0..5 {let(input,output)=layer_io(&self.features,&mut self.layers,l);k::teacher_layer(output.partition(layer_tile(l)),input,&self.weights[l],&self.biases[l]).generics(layer_generics(l)).compile()?;}
        k::quantize_cache_key((&mut self.keys).partition([32]),&self.features,self.width as i32,c.position_grid[0] as i32,c.position_grid[1] as i32,c.position_grid[2] as i32,c.view_bins as i32,c.light_bins as i32).compile()?;
        k::cache_lookup_nearest((&mut self.rgb_cache).partition([32,4]),&self.keys,self.cache.as_ref().ok_or_else(||anyhow::anyhow!("cache not prepared"))?,&self.layers[3],(c.entries()?/6) as i32,self.invalid_mask).compile()?;
        Ok(())
    }
    pub fn compile_interp(&mut self,c:&CacheConfig)->anyhow::Result<()> {
        k::cache_lookup_interp((&mut self.rgb_cache).partition([32,4]),&self.features,self.cache.as_ref().ok_or_else(||anyhow::anyhow!("cache not prepared"))?,self.width as i32,c.position_grid[0] as i32,c.position_grid[1] as i32,c.position_grid[2] as i32,c.view_bins as i32,c.light_bins as i32).compile()?;
        Ok(())
    }
    pub fn generate(&mut self,sequence:Sequence)->anyhow::Result<()> {
        // Buffers outlive finish(); single stream.
        unsafe{k::generate_frame_features((&mut self.features).partition([32,32]),&self.frame,self.width as i32,self.height as i32,sequence.code()).async_on(&self.stream)?;}
        Ok(())
    }
    pub fn teacher(&mut self)->anyhow::Result<()> {
        for l in 0..5 {
            let (input,output)=layer_io(&self.features,&mut self.layers,l);
            unsafe{k::teacher_layer(output.partition(layer_tile(l)),input,&self.weights[l],&self.biases[l])
                .generics(layer_generics(l)).async_on(&self.stream)?;}
        }Ok(())
    }
    pub fn full(&mut self,sequence:Sequence)->anyhow::Result<()> {
        self.generate(sequence)?;self.teacher()?;
        unsafe{k::advance((&mut self.frame).partition([1])).async_on(&self.stream)?;}
        Ok(())
    }
    /// Bytes of persistent device buffers, excluding allocator slack.
    pub fn allocated_bytes(&self)->usize {
        let n=self.n;let weights:usize=(0..5).map(|l|PADDED[l]*PADDED[l+1]+PADDED[l+1]).sum();
        4*(1+n*32+n*(128+128+64+32)+n+n*4+(n/32)*2+2+weights)+self.cache.as_ref().map_or(0,|_|self.cache_bytes)
    }
    pub fn finish(&self)->anyhow::Result<()> {unsafe{self.stream.synchronize()?;}Ok(())}
    pub fn read_rgb(&self)->anyhow::Result<Vec<f32>> {
        self.finish()?;
        let raw=api::dup(&self.layers[3]).to_host_vec().sync_on(&self.stream)?;
        Ok(raw.chunks_exact(32).take(self.width*self.height).flat_map(|p|p[..3].iter().copied()).collect())
    }
    pub fn read_cache_rgb(&self)->anyhow::Result<Vec<f32>> {
        self.finish()?;let raw=api::dup(&self.rgb_cache).to_host_vec().sync_on(&self.stream)?;
        Ok(raw.chunks_exact(4).take(self.width*self.height).flat_map(|p|p[..3].iter().copied()).collect())
    }
    pub fn lookup(&mut self,c:&CacheConfig)->anyhow::Result<()> {
        unsafe {
            k::quantize_cache_key((&mut self.keys).partition([32]),&self.features,self.width as i32,c.position_grid[0] as i32,c.position_grid[1] as i32,c.position_grid[2] as i32,c.view_bins as i32,c.light_bins as i32).async_on(&self.stream)?;
            k::cache_lookup_nearest((&mut self.rgb_cache).partition([32,4]),&self.keys,self.cache.as_ref().ok_or_else(||anyhow::anyhow!("cache not baked"))?,&self.layers[3],(c.entries()?/6) as i32,self.invalid_mask).async_on(&self.stream)?;
        }Ok(())
    }
    pub fn lookup_interp(&mut self,c:&CacheConfig)->anyhow::Result<()> {
        anyhow::ensure!(self.invalid_mask==0,"interpolated lookup has no per-object fallback yet");
        unsafe{k::cache_lookup_interp((&mut self.rgb_cache).partition([32,4]),&self.features,self.cache.as_ref().ok_or_else(||anyhow::anyhow!("cache not baked"))?,self.width as i32,c.position_grid[0] as i32,c.position_grid[1] as i32,c.position_grid[2] as i32,c.view_bins as i32,c.light_bins as i32).async_on(&self.stream)?;}
        Ok(())
    }
    pub fn cached_interp(&mut self,sequence:Sequence,c:&CacheConfig)->anyhow::Result<()> {
        self.generate(sequence)?;self.lookup_interp(c)?;
        unsafe{k::advance((&mut self.frame).partition([1])).async_on(&self.stream)?;}Ok(())
    }
    pub fn cached(&mut self,sequence:Sequence,c:&CacheConfig)->anyhow::Result<()> {
        self.generate(sequence)?;if self.invalid_mask!=0 {self.teacher()?;}self.lookup(c)?;
        unsafe{k::advance((&mut self.frame).partition([1])).async_on(&self.stream)?;}Ok(())
    }
    pub fn bake(&mut self,c:&CacheConfig,teacher:&Teacher)->anyhow::Result<()> {
        let entries=c.entries()?;
        let cache:Tensor<f32>=api::zeros(&[entries,4]).sync_on(&self.stream)?;
        let mut batch=Gpu::new(32768,1,teacher)?;
        for start in (0..entries).step_by(batch.n) {
            unsafe{k::generate_bake_features((&mut batch.features).partition([32,32]),start as i32,c.position_grid[0] as i32,c.position_grid[1] as i32,c.position_grid[2] as i32,c.view_bins as i32,c.light_bins as i32).async_on(&batch.stream)?;}
            batch.teacher()?;
            unsafe {
                k::pack_rgb((&mut batch.rgb_cache).partition([32,4]),&batch.layers[3]).async_on(&batch.stream)?;
                // Disjoint in-bounds range, both buffers live until finish().
                cuda_core::memcpy_dtod_async::<f32>(cache.device_pointer().cu_deviceptr()+(start*16) as u64,batch.rgb_cache.device_pointer().cu_deviceptr(),(entries-start).min(batch.n)*4,&batch.stream)?;
            }
        }
        batch.finish()?;self.cache=Some(cache);self.cache_bytes=entries*16;Ok(())
    }
    pub fn invalidate_object(&mut self,id:usize)->anyhow::Result<()> {anyhow::ensure!(id<6,"invalid object id");self.finish()?;self.graph=None;self.full_graph=None;self.invalid_mask|=1<<id;Ok(())}
    pub fn capture(&mut self,sequence:Sequence,c:&CacheConfig)->anyhow::Result<()> {
        anyhow::ensure!(self.invalid_mask==0,"rebuild invalidated namespaces before graph capture");
        self.cached(sequence,c)?;self.finish()?;
        let cache=self.cache.as_ref().ok_or_else(||anyhow::anyhow!("no cache"))?;
        let per_object=(c.entries()?/6) as i32;
        let graph=CudaGraph::scope(&self.stream,|s| {
            s.record(k::generate_frame_features((&mut self.features).partition([32,32]),&self.frame,self.width as i32,self.height as i32,sequence.code()))?;
            s.record(k::quantize_cache_key((&mut self.keys).partition([32]),&self.features,self.width as i32,c.position_grid[0] as i32,c.position_grid[1] as i32,c.position_grid[2] as i32,c.view_bins as i32,c.light_bins as i32))?;
            s.record(k::cache_lookup_nearest((&mut self.rgb_cache).partition([32,4]),&self.keys,cache,&self.layers[3],per_object,0))?;
            s.record(k::advance((&mut self.frame).partition([1])))?;
            Ok(())
        })?;
        self.graph=Some(graph);Ok(())
    }
    /// FULL frame graph: features, five teacher layers, frame advance.
    pub fn capture_full(&mut self,sequence:Sequence)->anyhow::Result<()> {
        self.full(sequence)?;self.finish()?;
        let (w,h)=(self.width as i32,self.height as i32);
        let features=&mut self.features;let layers=&mut self.layers;let frame=&mut self.frame;let (weights,biases)=(&self.weights,&self.biases);
        let graph=CudaGraph::scope(&self.stream,|s| {
            s.record(k::generate_frame_features(features.partition([32,32]),&*frame,w,h,sequence.code()))?;
            for l in 0..5 {
                let (input,output)=layer_io(features,layers,l);
                s.record(k::teacher_layer(output.partition(layer_tile(l)),input,&weights[l],&biases[l]).generics(layer_generics(l)))?;
            }
            s.record(k::advance(frame.partition([1])))?;
            Ok(())
        })?;
        self.full_graph=Some(graph);Ok(())
    }
    pub fn replay_full(&mut self)->anyhow::Result<()> {
        let graph=self.full_graph.as_ref().ok_or_else(||anyhow::anyhow!("FULL graph not captured"))?;
        unsafe{graph.launch().async_on(&self.stream)?;}Ok(())
    }
    pub fn replay(&mut self)->anyhow::Result<()> {
        let graph=self.graph.as_ref().ok_or_else(||anyhow::anyhow!("graph not captured"))?;
        // Captured buffers are owned by self.
        unsafe{graph.launch().async_on(&self.stream)?;}Ok(())
    }
    pub fn save_cache(&self,path:&std::path::Path,c:&CacheConfig,teacher:&Teacher)->anyhow::Result<()> {
        self.finish()?;let data=api::dup(self.cache.as_ref().ok_or_else(||anyhow::anyhow!("no cache"))?).to_host_vec().sync_on(&self.stream)?;
        let mut meta=narc_core::persistence::Metadata::new(c,teacher.hash())?;
        for i in 0..6 {if self.invalid_mask&(1<<i)!=0 {meta.invalidate(i)?;}}
        narc_core::persistence::save(path,&meta,&data)
    }
    pub fn load_cache(&mut self,path:&std::path::Path,c:&CacheConfig,teacher:&Teacher)->anyhow::Result<()> {
        let expected=narc_core::persistence::Metadata::new(c,teacher.hash())?;
        let (meta,data)=narc_core::persistence::load(path,&expected)?;
        self.cache=Some(upload(data,&[c.entries()?,4],&self.stream)?);self.invalid_mask=0;self.cache_bytes=c.bytes()?;
        for i in 0..6{if !meta.valid_objects[i]{self.invalid_mask|=1<<i;}}Ok(())
    }
    pub fn hit_count(&self)->anyhow::Result<usize>{self.finish()?;let raw=api::dup(&self.rgb_cache).to_host_vec().sync_on(&self.stream)?;Ok(raw.chunks_exact(4).take(self.width*self.height).filter(|p|p[3]==1.).count())}
    pub fn error_sums(&mut self)->anyhow::Result<(f64,f64)> {
        unsafe{k::error_rgb((&mut self.errors).partition([1,2]),&self.layers[3],&self.rgb_cache).async_on(&self.stream)?;}
        k::reduce_mse((&mut self.reduced).partition([1,2]),&self.errors,(self.n/32).div_ceil(256) as i32).sync_on(&self.stream)?;
        let values=api::dup(&self.reduced).to_host_vec().sync_on(&self.stream)?;
        Ok((values[0] as f64,values[1] as f64))
    }
    /// Per-frame (sum of squared error, max abs error) of each cached path vs FULL.
    /// One readback at the end.
    pub fn sequence_quality(&mut self,frames:usize,sequence:Sequence,paths:usize,mut run:impl FnMut(&mut Gpu,usize)->anyhow::Result<()>)->anyhow::Result<Vec<Vec<(f64,f64)>>> {
        anyhow::ensure!(self.invalid_mask==0,"quality pass requires a fully valid cache");
        if paths==0 {return Ok(Vec::new());}
        let stats:Tensor<f32>=api::zeros(&[paths*frames,2]).sync_on(&self.stream)?;
        for i in 0..frames {
            self.reset(i as i32)?;self.full(sequence)?;
            for p in 0..paths {
                self.reset(i as i32)?;run(self,p)?;
                unsafe{
                    k::error_rgb((&mut self.errors).partition([1,2]),&self.layers[3],&self.rgb_cache).async_on(&self.stream)?;
                    k::reduce_mse((&mut self.reduced).partition([1,2]),&self.errors,(self.n/32).div_ceil(256) as i32).async_on(&self.stream)?;
                    // Two f32 per (path, frame), same stream.
                    cuda_core::memcpy_dtod_async::<f32>(stats.device_pointer().cu_deviceptr()+((p*frames+i)*8) as u64,self.reduced.device_pointer().cu_deviceptr(),2,&self.stream)?;
                }
            }
        }
        self.finish()?;let data=stats.to_host_vec().sync_on(&self.stream)?;
        Ok(data.chunks_exact(2).map(|p|(p[0] as f64,p[1] as f64)).collect::<Vec<_>>().chunks(frames).map(|c|c.to_vec()).collect())
    }
    /// Fraction of samples whose key is unchanged since the previous frame.
    pub fn key_reuse(&mut self,frames:usize,sequence:Sequence,c:&CacheConfig)->anyhow::Result<Vec<f64>> {
        if frames<2 {return Ok(Vec::new());}
        let stats:Tensor<f32>=api::zeros(&[frames,2]).sync_on(&self.stream)?;
        let prev:Tensor<i32>=api::zeros(&[self.n]).sync_on(&self.stream)?;
        for i in 0..frames {
            self.reset(i as i32)?;self.generate(sequence)?;self.lookup(c)?;
            unsafe{
                if i>0 {
                    k::key_match((&mut self.errors).partition([1,2]),&self.keys,&prev).async_on(&self.stream)?;
                    k::reduce_mse((&mut self.reduced).partition([1,2]),&self.errors,(self.n/32).div_ceil(256) as i32).async_on(&self.stream)?;
                    cuda_core::memcpy_dtod_async::<f32>(stats.device_pointer().cu_deviceptr()+(i*8) as u64,self.reduced.device_pointer().cu_deviceptr(),2,&self.stream)?;
                }
                // Same-size buffers, same stream.
                cuda_core::memcpy_dtod_async::<i32>(prev.device_pointer().cu_deviceptr(),self.keys.device_pointer().cu_deviceptr(),self.n,&self.stream)?;
            }
        }
        self.finish()?;let data=stats.to_host_vec().sync_on(&self.stream)?;
        let pixels=(self.width*self.height) as f64;
        Ok(data.chunks_exact(2).skip(1).map(|p|p[0] as f64/pixels).collect())
    }
}
impl Drop for Gpu {fn drop(&mut self){let _=self.finish();}}
