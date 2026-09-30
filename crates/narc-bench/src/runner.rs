use std::{path::Path,time::Instant};
use serde::{Serialize,Deserialize};
use cuda_core::IntoResult;
use narc_core::{config::Config,teacher::Teacher,metrics::{ErrorMetrics,quantile}};
use narc_gpu::Gpu;
use crate::hot_loop;

#[derive(Clone,Copy,Debug,Serialize,Deserialize,PartialEq,Eq)]
#[serde(rename_all="SCREAMING_SNAKE_CASE")]
pub enum Mode {Full,FullGraph,Cache,CacheInterp,CacheGraph}
impl Mode {
    pub fn name(self)->&'static str{match self{Self::Full=>"FULL",Self::FullGraph=>"FULL_GRAPH",Self::Cache=>"CACHE_ONLY",Self::CacheInterp=>"CACHE_INTERP",Self::CacheGraph=>"CACHE_GRAPH"}}
    pub fn cached(self)->bool{matches!(self,Self::Cache|Self::CacheInterp|Self::CacheGraph)}
    pub fn interpolation(self)->&'static str{match self{Self::Cache|Self::CacheGraph=>"nearest",Self::CacheInterp=>"multilinear",_=>"none"}}
}
impl std::str::FromStr for Mode {type Err=anyhow::Error;fn from_str(s:&str)->anyhow::Result<Self>{match s.to_lowercase().replace('_',"-").as_str(){
    "full"=>Ok(Self::Full),"full-graph"=>Ok(Self::FullGraph),"cache"|"cache-only"|"cache-nearest"=>Ok(Self::Cache),"cache-interp"|"interp"=>Ok(Self::CacheInterp),"cache-graph"=>Ok(Self::CacheGraph),
    "narc"|"narc-graph"=>anyhow::bail!("{s}: the residual network is not implemented, so NARC would be identical to CACHE_ONLY; use cache, cache-interp or cache-graph"),
    _=>anyhow::bail!("unknown mode {s}")}}}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Environment {pub gpu_name:String,pub compute_capability:String,pub driver_cuda_version:i32,pub cuda_toolkit:String,pub rust_version:String,pub cutile_version:String,pub git_commit:String,pub total_vram_bytes:usize}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Record {
    pub gpu_name:String,pub compute_capability:String,pub cuda_version:i32,pub rust_version:String,pub cutile_version:String,pub git_commit:String,
    pub config:String,pub resolution:String,pub sequence:String,pub mode:String,pub interpolation:String,
    pub position_grid:String,pub view_bins:usize,pub light_bins:usize,
    pub frame_count:usize,pub median_gpu_ms:f64,pub p95_gpu_ms:f64,pub min_gpu_ms:f64,pub mean_gpu_ms:f64,pub equivalent_fps:f64,
    pub latency_frame_count:usize,pub latency_median_ms:f64,pub latency_p95_ms:f64,
    pub speedup_vs_full:Option<f64>,pub latency_speedup_vs_full:Option<f64>,
    pub cache_hit_rate:Option<f64>,pub temporal_key_reuse:Option<f64>,
    pub cache_size_bytes:Option<usize>,pub working_set_estimate_bytes:usize,#[serde(default)] pub allocated_vram_bytes:usize,pub measured_vram_bytes:usize,
    pub mse:f64,pub rmse:f64,pub psnr_db:Option<f64>,pub worst_frame_psnr_db:Option<f64>,pub max_abs_error:f64,pub quality_frame_count:usize,
    pub cold_compile_ms:f64,pub jit_compiles:u64,pub jit_backend_compiles:u64,pub jit_disk_hits:u64,pub bake_ms:f64,pub warm_start_ms:f64,
    pub graph_matches_eager:Option<bool>,pub residual_enabled:bool,
}
#[derive(Clone,Serialize,Deserialize)]
pub struct QualitySeries {pub mode:String,pub per_frame:Vec<ErrorMetrics>}
#[derive(Clone,Serialize,Deserialize)]
pub struct RunResult {pub config_name:String,pub config:Config,pub modes:Vec<String>,pub pipelined_ms:Vec<Vec<f64>>,pub latency_ms:Vec<Vec<f64>>,pub quality:Vec<QualitySeries>,pub key_reuse_per_frame:Vec<f64>}
#[derive(Clone,Serialize,Deserialize)]
pub struct Report {pub environment:Environment,pub notes:Vec<String>,pub records:Vec<Record>,pub runs:Vec<RunResult>}

pub fn notes()->Vec<String>{vec![
    "Pipeline-equivalent FPS = 1000 / median GPU ms of the synthetic neural pipeline; not game FPS, not DLSS.".into(),
    "median/p95_gpu_ms: pipelined timing. Frames are enqueued in batches of 32 without host waits; one CUDA event pair brackets each frame pipeline, so the value is device execution time.".into(),
    "latency_*: one event pair per frame after a host wait, so host launch overhead is visible. CUDA Graph replay targets this regime.".into(),
    "Modes are interleaved in rotating order at identical frame IDs. JIT is warmed first; a JIT during measurement aborts the run.".into(),
    "The residual network is not implemented. There is no NARC mode; CACHE_ONLY, CACHE_INTERP and CACHE_GRAPH are pure cache paths.".into(),
    "cache_hit_rate: fraction of samples served from a valid cache namespace. The dense cache is exhaustive over the scene domain, so it is 1.0 by construction and says nothing about quality.".into(),
    "temporal_key_reuse: fraction of samples whose nearest cache key equals the previous frame key. This measures actual reuse under camera/light motion.".into(),
    "Quality: MSE, RMSE, PSNR (peak 1.0) and max abs error of RGB against FULL, accumulated over every measured frame ID. PSNR null means identical images.".into(),
    "allocated_vram_bytes: exact bytes of the persistent buffers (frame buffers, weights, cache). measured_vram_bytes: device free-memory delta from before allocation to ready-to-measure, allocator reservation included; valid only for the first session of a process, since freed buffers stay in the async memory pool.".into(),
    "cold_compile_ms: runtime kernel JIT wall time in this process; bake_ms includes bake kernel JIT.".into(),
]}

pub fn command_version(program:&str,args:&[&str])->String{std::process::Command::new(program).args(args).output().ok().filter(|o|o.status.success()).map(|o|String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_else(||"unavailable".into())}
pub fn memory_available()->anyhow::Result<(usize,usize)>{let mut free=0;let mut total=0;unsafe{cuda_core::sys::cuMemGetInfo_v2(&mut free,&mut total).result()?;}Ok((free,total))}
pub fn environment()->anyhow::Result<Environment>{
    let d=cuda_core::Device::new(0)?;d.bind_to_thread()?;
    let mut driver=0;unsafe{cuda_core::sys::cuDriverGetVersion(&mut driver).result()?;}
    let nvcc=command_version("nvcc",&["--version"]);
    Ok(Environment{gpu_name:d.name()?,compute_capability:unsafe{cuda_core::get_device_sm_name(d.cu_device())?},driver_cuda_version:driver,
        cuda_toolkit:nvcc.lines().last().unwrap_or("unavailable").to_string(),rust_version:command_version("rustc",&["--version"]),cutile_version:"0.3.1".into(),
        git_commit:command_version("git",&["describe","--always","--dirty"]),total_vram_bytes:memory_available()?.1})
}

/// GPU state for one resolution: cache and compiled kernels.
pub struct Session {pub gpu:Gpu,pub teacher:Teacher,pub bake_ms:f64,pub cold_compile_ms:f64,pub jit:(u64,u64,u64),pub vram_used:usize,pub startup:Instant}
pub fn prepare(c:&Config,cache_path:Option<&Path>,interp:bool)->anyhow::Result<Session> {
    let startup=Instant::now();
    c.validate()?;let d=cuda_core::Device::new(0)?;d.bind_to_thread()?;
    let(free,_)=memory_available()?;let estimate=c.working_set_bytes()?;
    println!("Cache {:.1} MiB; estimated working set {:.2} GiB; free VRAM {:.2} GiB",c.cache.bytes()? as f64/1048576.,estimate as f64/1073741824.,free as f64/1073741824.);
    anyhow::ensure!(estimate+128*1024*1024<free,"insufficient free VRAM for persistent buffers and reserve");
    let teacher=Teacher::new(c.teacher.seed);let mut gpu=Gpu::new(c.render.width,c.render.height,&teacher)?;
    let start=Instant::now();if let Some(p)=cache_path{gpu.load_cache(p,&c.cache,&teacher)?;}else{gpu.bake(&c.cache,&teacher)?;}gpu.finish()?;let bake_ms=start.elapsed().as_secs_f64()*1000.;
    let compile=Instant::now();gpu.compile_runtime(c.render.sequence,&c.cache)?;if interp{gpu.compile_interp(&c.cache)?;}let cold_compile_ms=compile.elapsed().as_secs_f64()*1000.;
    let jit=(cutile::tile_kernel::jit_compile_count(),cutile::jit_cache::jit_backend_compile_count(),cutile::jit_cache::jit_disk_hit_count());
    gpu.finish()?;let vram_used=free.saturating_sub(memory_available()?.0);
    Ok(Session{gpu,teacher,bake_ms,cold_compile_ms,jit,vram_used,startup})
}

fn graph_matches(gpu:&mut Gpu,c:&Config,eager:hot_loop::Launch,graph:hot_loop::Launch,full:bool)->anyhow::Result<bool> {
    let frame=(c.render.frames.saturating_sub(1)).min(7) as i32;
    let read=|g:&Gpu|if full{g.read_rgb()}else{g.read_cache_rgb()};
    gpu.reset(frame)?;eager(gpu,c)?;let a=read(gpu)?;
    gpu.reset(frame)?;graph(gpu,c)?;let b=read(gpu)?;
    Ok(a==b)
}

/// Measures every mode on one sequence.
pub fn run_sequence(s:&mut Session,c:&Config,config_name:&str,modes:&[Mode],latency_frames:usize,env:&Environment)->anyhow::Result<(Vec<Record>,RunResult)> {
    anyhow::ensure!(!cfg!(debug_assertions),"benchmark requires --release");anyhow::ensure!(!modes.is_empty(),"select at least one mode");
    let seq=c.render.sequence;let gpu=&mut s.gpu;
    let mut cache_graph_ok=None;let mut full_graph_ok=None;
    if modes.contains(&Mode::CacheGraph){gpu.capture(seq,&c.cache)?;cache_graph_ok=Some(graph_matches(gpu,c,|g,c|g.cached(c.render.sequence,&c.cache),|g,_|g.replay(),false)?);}
    if modes.contains(&Mode::FullGraph){gpu.capture_full(seq)?;full_graph_ok=Some(graph_matches(gpu,c,|g,c|g.full(c.render.sequence),|g,_|g.replay_full(),true)?);}
    for i in 0..c.benchmark.warmup_frames.max(1){for &m in modes{gpu.reset(i as i32)?;hot_loop::launch(gpu,c,m)?;}}
    gpu.finish()?;let warm_start_ms=s.startup.elapsed().as_secs_f64()*1000.;
    println!("[{} {}x{} {:?}] timing {} pipelined + {} latency frames across {} modes",config_name,c.render.width,c.render.height,seq,c.render.frames,latency_frames,modes.len());
    let t=hot_loop::measure(gpu,c,modes,latency_frames)?;
    let cached:Vec<Mode>=modes.iter().copied().filter(|m|m.cached()).collect();
    println!("[{} {}x{} {:?}] quality over all {} frame IDs for {} cached modes",config_name,c.render.width,c.render.height,seq,c.render.frames,cached.len());
    let sums=gpu.sequence_quality(c.render.frames,seq,cached.len(),|g,p|hot_loop::launch(g,c,cached[p]))?;
    let mut hits=Vec::new();for &m in &cached{gpu.reset(c.render.frames as i32-1)?;hot_loop::launch(gpu,c,m)?;hits.push(gpu.hit_count()? as f64/(c.render.width*c.render.height) as f64);}
    let reuse=gpu.key_reuse(c.render.frames,seq,&c.cache)?;
    let mean_reuse=if reuse.is_empty(){None}else{Some(reuse.iter().sum::<f64>()/reuse.len() as f64)};
    let count=c.render.width*c.render.height*3;
    let full_idx=modes.iter().position(|&m|m==Mode::Full);
    let med=|v:&[f64]|quantile(v,0.5);
    let mut records=Vec::new();let mut quality=Vec::new();
    for (i,&mode) in modes.iter().enumerate(){
        let (err,worst,hit)=match cached.iter().position(|&m|m==mode) {
            Some(p)=>{
                let per:Vec<ErrorMetrics>=sums[p].iter().map(|&(e,m)|ErrorMetrics::from_sums(e,m,count)).collect();
                let total=ErrorMetrics::from_sums(sums[p].iter().map(|x|x.0).sum(),sums[p].iter().map(|x|x.1).fold(0.,f64::max),count*c.render.frames);
                let worst=per.iter().filter_map(|e|e.psnr_db).fold(None,|a:Option<f64>,b|Some(a.map_or(b,|a|a.min(b))));
                quality.push(QualitySeries{mode:mode.name().into(),per_frame:per});(total,worst,Some(hits[p]))
            }
            None=>(ErrorMetrics::from_sums(0.,0.,count),None,None),
        };
        let p=&t.pipelined[i];let l=&t.latency[i];
        records.push(Record{gpu_name:env.gpu_name.clone(),compute_capability:env.compute_capability.clone(),cuda_version:env.driver_cuda_version,rust_version:env.rust_version.clone(),cutile_version:env.cutile_version.clone(),git_commit:env.git_commit.clone(),
            config:config_name.into(),resolution:format!("{}x{}",c.render.width,c.render.height),sequence:format!("{:?}",seq),mode:mode.name().into(),interpolation:mode.interpolation().into(),
            position_grid:format!("{}x{}x{}",c.cache.position_grid[0],c.cache.position_grid[1],c.cache.position_grid[2]),view_bins:c.cache.view_bins,light_bins:c.cache.light_bins,
            frame_count:p.len(),median_gpu_ms:med(p),p95_gpu_ms:quantile(p,0.95),min_gpu_ms:p.iter().copied().fold(f64::INFINITY,f64::min),mean_gpu_ms:p.iter().sum::<f64>()/p.len() as f64,equivalent_fps:1000./med(p),
            latency_frame_count:l.len(),latency_median_ms:if l.is_empty(){f64::NAN}else{med(l)},latency_p95_ms:if l.is_empty(){f64::NAN}else{quantile(l,0.95)},
            speedup_vs_full:full_idx.map(|f|med(&t.pipelined[f])/med(p)),latency_speedup_vs_full:full_idx.filter(|_|!l.is_empty()).map(|f|med(&t.latency[f])/med(l)),
            cache_hit_rate:hit,temporal_key_reuse:if mode.cached(){mean_reuse}else{None},
            cache_size_bytes:if mode.cached(){Some(c.cache.bytes()?)}else{None},working_set_estimate_bytes:c.working_set_bytes()?,allocated_vram_bytes:s.gpu.allocated_bytes(),measured_vram_bytes:s.vram_used,
            mse:err.mse,rmse:err.rmse,psnr_db:err.psnr_db,worst_frame_psnr_db:worst,max_abs_error:err.max_abs_error,quality_frame_count:if mode.cached(){c.render.frames}else{0},
            cold_compile_ms:s.cold_compile_ms,jit_compiles:s.jit.0,jit_backend_compiles:s.jit.1,jit_disk_hits:s.jit.2,bake_ms:s.bake_ms,warm_start_ms,
            graph_matches_eager:match mode{Mode::CacheGraph=>cache_graph_ok,Mode::FullGraph=>full_graph_ok,_=>None},residual_enabled:false});
    }
    let run=RunResult{config_name:config_name.into(),config:c.clone(),modes:modes.iter().map(|m|m.name().to_string()).collect(),pipelined_ms:t.pipelined,latency_ms:t.latency,quality,key_reuse_per_frame:reuse};
    Ok((records,run))
}

pub fn write_report(out:&Path,report:&Report)->anyhow::Result<()> {
    std::fs::create_dir_all(out)?;std::fs::write(out.join("benchmark.json"),serde_json::to_vec_pretty(report)?)?;
    let mut csv=csv::Writer::from_path(out.join("benchmark.csv"))?;for r in &report.records{csv.serialize(r)?;}csv.flush()?;Ok(())
}

fn size(bytes:usize)->String{if bytes>=1<<30{format!("{:.2} GB",bytes as f64/1073741824.)}else{format!("{:.1} MB",bytes as f64/1048576.)}}
fn psnr(p:Option<f64>)->String{p.map_or("identical".into(),|v|format!("{v:.2} dB"))}
/// Console summary for one (config, resolution, sequence) group.
pub fn summary(env:&Environment,records:&[Record])->String {
    use std::fmt::Write;let mut s=String::new();let Some(r0)=records.first() else{return s;};
    let _=writeln!(s,"\nNARC benchmark - {} - {} - {} - config {} (grid {}, view {}, light {})",env.gpu_name.trim_start_matches("NVIDIA GeForce "),r0.resolution,r0.sequence,r0.config,r0.position_grid,r0.view_bins,r0.light_bins);
    for r in records {
        let _=writeln!(s,"\n{}\nmedian GPU:       {:>9.3} ms\np95 GPU:          {:>9.3} ms\nequiv pipeline:   {:>9.1} fps",r.mode,r.median_gpu_ms,r.p95_gpu_ms,r.equivalent_fps);
        if r.mode!="FULL" {
            if let Some(x)=r.speedup_vs_full{let _=writeln!(s,"speedup vs FULL:  {x:>9.2}x");}
            let _=writeln!(s,"PSNR vs FULL:     {:>12}\nworst frame PSNR: {:>12}\nMSE:              {:>12.3e}\nRMSE:             {:>12.5}\nmax abs error:    {:>12.5}",psnr(r.psnr_db),psnr(r.worst_frame_psnr_db),r.mse,r.rmse,r.max_abs_error);
            if let Some(h)=r.cache_hit_rate{let _=writeln!(s,"cache hit:        {:>9.1} %  (valid namespace, not a quality measure)",h*100.);}
            if let Some(u)=r.temporal_key_reuse{let _=writeln!(s,"key reuse f/f-1:  {:>9.1} %",u*100.);}
            if let Some(b)=r.cache_size_bytes{let _=writeln!(s,"cache VRAM:       {:>12}",size(b));}
            if let Some(g)=r.graph_matches_eager{let _=writeln!(s,"graph == eager:   {g:>12}");}
        }
    }
    let lat:Vec<String>=records.iter().map(|r|format!("{} {:.3} ms",r.mode,r.latency_median_ms)).collect();
    let _=writeln!(s,"\nlatency mode (host wait per frame, median): {}",lat.join(", "));
    let _=writeln!(s,"allocated VRAM: {} ; measured VRAM delta: {} ; cold compile {:.0} ms ({} JIT, {} backend, {} disk hits) ; bake {:.0} ms",size(r0.allocated_vram_bytes),size(r0.measured_vram_bytes),r0.cold_compile_ms,r0.jit_compiles,r0.jit_backend_compiles,r0.jit_disk_hits,r0.bake_ms);
    s
}
