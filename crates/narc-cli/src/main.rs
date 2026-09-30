mod game;
use std::path::{Path,PathBuf};
use clap::{Args,Parser,Subcommand};
use narc_bench::runner::{self,Mode,Report};
use narc_core::{config::Config,metrics::ErrorMetrics,scene::Sequence};

#[derive(Parser)]
#[command(name="narc",about="NARC: neural adaptive rendering cache, synthetic GPU benchmark (not DLSS)")]
struct Cli {#[command(subcommand)] cmd:Cmd}

#[derive(Subcommand)]
enum Cmd {
    /// Print device, toolchain and cache-size estimates.
    Info {#[command(flatten)] cfg:CfgArgs},
    /// Bake the dense cache on the GPU and save it to disk.
    Bake {#[command(flatten)] cfg:CfgArgs,#[arg(long,default_value="out/cache/default.narc")] out:PathBuf},
    /// Measure FULL against the cache paths and write JSON/CSV.
    Bench(BenchArgs),
    /// Export FULL, cache and error heatmap images for one frame.
    Render {#[command(flatten)] cfg:CfgArgs,#[arg(long,default_value="camera")] sequence:Sequence,#[arg(long,default_value_t=120)] frame:u32,#[arg(long,default_value="out/render")] out:PathBuf},
    /// Static scan of the measured path.
    VerifyZeroCopy,
    /// Markdown tables from benchmark.json files.
    Report {files:Vec<PathBuf>},
    /// Offline pipeline on game captures.
    Game {#[command(subcommand)] cmd:game::GameCmd},
}

#[derive(Args,Clone)]
struct CfgArgs {
    #[arg(long,default_value="configs/default.toml")] config:PathBuf,
    /// WIDTHxHEIGHT; overrides the config.
    #[arg(long)] resolution:Option<String>,
    /// Cubic position grid; overrides the config.
    #[arg(long)] grid:Option<usize>,
    #[arg(long)] view_bins:Option<usize>,
    #[arg(long)] light_bins:Option<usize>,
}

#[derive(Args)]
struct BenchArgs {
    #[arg(long,value_delimiter=',',default_value="configs/default.toml")] configs:Vec<PathBuf>,
    /// Comma-separated WIDTHxHEIGHT list; defaults to the config resolution.
    #[arg(long,value_delimiter=',')] resolution:Vec<String>,
    /// static, camera, camera-light or all.
    #[arg(long,value_delimiter=',',default_value="all")] sequence:Vec<String>,
    #[arg(long,value_delimiter=',',default_value="full,full-graph,cache,cache-interp,cache-graph")] modes:Vec<Mode>,
    /// Measured (pipelined) frames; also the number of frame IDs checked for quality.
    #[arg(long)] frames:Option<usize>,
    #[arg(long,default_value_t=100)] latency_frames:usize,
    #[arg(long)] warmup:Option<usize>,
    #[arg(long)] grid:Option<usize>,
    #[arg(long)] view_bins:Option<usize>,
    #[arg(long)] light_bins:Option<usize>,
    /// Load a baked cache instead of baking (must match the config).
    #[arg(long)] cache:Option<PathBuf>,
    /// Persist compiled cubins across processes (cuTile disk cache).
    #[arg(long)] jit_disk_cache:bool,
    #[arg(long,default_value="out/bench")] out:PathBuf,
}

fn parse_resolution(s:&str)->anyhow::Result<(usize,usize)>{let (w,h)=s.split_once('x').ok_or_else(||anyhow::anyhow!("resolution must be WIDTHxHEIGHT"))?;Ok((w.trim().parse()?,h.trim().parse()?))}
fn apply(c:&mut Config,grid:Option<usize>,view:Option<usize>,light:Option<usize>){if let Some(g)=grid{c.cache.position_grid=[g;3];}if let Some(v)=view{c.cache.view_bins=v;}if let Some(l)=light{c.cache.light_bins=l;}}
fn load(a:&CfgArgs)->anyhow::Result<Config>{
    let mut c=Config::load(&a.config)?;apply(&mut c,a.grid,a.view_bins,a.light_bins);
    if let Some(r)=&a.resolution{let(w,h)=parse_resolution(r)?;c.render.width=w;c.render.height=h;}
    c.validate()?;Ok(c)
}
fn config_name(path:&Path,c:&Config,overridden:bool)->String{
    let stem=path.file_stem().and_then(|s|s.to_str()).unwrap_or("config").to_string();
    if overridden{format!("{stem}-g{}v{}l{}",c.cache.position_grid[0],c.cache.view_bins,c.cache.light_bins)}else{stem}
}

fn main()->anyhow::Result<()> {
    match Cli::parse().cmd {
        Cmd::Info{cfg}=>info(&cfg),
        Cmd::Bake{cfg,out}=>bake(&cfg,&out),
        Cmd::Bench(a)=>bench(a),
        Cmd::Render{cfg,sequence,frame,out}=>render(&cfg,sequence,frame,&out),
        Cmd::VerifyZeroCopy=>verify_zero_copy(),
        Cmd::Report{files}=>report(&files),
        Cmd::Game{cmd}=>game::run(cmd),
    }
}

fn report(files:&[PathBuf])->anyhow::Result<()> {
    let mut records=Vec::new();let mut env=None;
    for f in files{let r:Report=serde_json::from_slice(&std::fs::read(f)?)?;env.get_or_insert(r.environment);records.extend(r.records);}
    let env=env.ok_or_else(||anyhow::anyhow!("no benchmark.json given"))?;
    let mb=|b:Option<usize>|b.map_or("-".into(),|b|format!("{:.1}",b as f64/1048576.));
    let db=|p:Option<f64>|p.map_or("identical".into(),|v|format!("{v:.2}"));
    let opt=|v:Option<f64>,d:usize|v.map_or("-".into(),|v|format!("{v:.d$}"));
    println!("Environment: {} ({}), driver CUDA {}, {}, {}, cuTile {}, commit {}\n",env.gpu_name,env.compute_capability,env.driver_cuda_version,env.cuda_toolkit,env.rust_version,env.cutile_version,env.git_commit);
    println!("| config | resolution | sequence | mode | median GPU ms | p95 GPU ms | pipeline fps | speedup vs FULL | latency ms | PSNR dB | worst frame dB | RMSE | max abs err | hit rate | key reuse | cache MB |");
    println!("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for r in &records {
        let q=r.mode!="FULL"&&r.mode!="FULL_GRAPH";
        println!("| {} | {} | {} | {} | {:.3} | {:.3} | {:.1} | {} | {:.3} | {} | {} | {} | {} | {} | {} | {} |",r.config,r.resolution,r.sequence,r.mode,r.median_gpu_ms,r.p95_gpu_ms,r.equivalent_fps,opt(r.speedup_vs_full,2),r.latency_median_ms,
            if q{db(r.psnr_db)}else{"ref".into()},if q{db(r.worst_frame_psnr_db)}else{"-".into()},if q{format!("{:.5}",r.rmse)}else{"-".into()},if q{format!("{:.4}",r.max_abs_error)}else{"-".into()},
            opt(r.cache_hit_rate.map(|h|h*100.),1),opt(r.temporal_key_reuse.map(|h|h*100.),1),mb(r.cache_size_bytes));
    }
    println!("\nCUDA Graph, eager launches vs replay (median ms):\n\n| config | resolution | sequence | path | eager pipelined | graph pipelined | eager latency | graph latency | graph == eager |\n|---|---|---|---|---:|---:|---:|---:|---|");
    for (eager,graph,path) in [("FULL","FULL_GRAPH","FULL"),("CACHE_ONLY","CACHE_GRAPH","CACHE nearest")] {
        for g in records.iter().filter(|r|r.mode==graph) {
            if let Some(e)=records.iter().find(|r|r.mode==eager&&r.config==g.config&&r.resolution==g.resolution&&r.sequence==g.sequence) {
                println!("| {} | {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {} |",g.config,g.resolution,g.sequence,path,e.median_gpu_ms,g.median_gpu_ms,e.latency_median_ms,g.latency_median_ms,g.graph_matches_eager.map_or("-".into(),|b|b.to_string()));
            }
        }
    }
    let mut cached:Vec<&runner::Record>=records.iter().filter(|r|r.mode=="CACHE_ONLY"||r.mode=="CACHE_INTERP").collect();
    cached.sort_by(|a,b|a.cache_size_bytes.cmp(&b.cache_size_bytes).then(a.resolution.cmp(&b.resolution)).then(a.sequence.cmp(&b.sequence)).then(a.mode.cmp(&b.mode)));
    println!("\nSpeed x quality x memory, sorted by cache size (no ranking heuristic):\n\n| cache MB | grid | view | light | resolution | sequence | lookup | median GPU ms | speedup vs FULL | PSNR dB | worst frame dB | max abs err | key reuse |\n|---:|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|");
    for r in cached {
        println!("| {} | {} | {} | {} | {} | {} | {} | {:.3} | {} | {} | {} | {:.4} | {} |",mb(r.cache_size_bytes),r.position_grid,r.view_bins,r.light_bins,r.resolution,r.sequence,r.interpolation,r.median_gpu_ms,opt(r.speedup_vs_full,1),db(r.psnr_db),db(r.worst_frame_psnr_db),r.max_abs_error,opt(r.temporal_key_reuse.map(|h|h*100.),1));
    }
    Ok(())
}

fn info(a:&CfgArgs)->anyhow::Result<()> {
    let env=runner::environment()?;println!("{}",serde_json::to_string_pretty(&env)?);
    let (free,_)=runner::memory_available()?;println!("free VRAM: {:.2} GiB",free as f64/1073741824.);
    let c=load(a)?;
    println!("config {}: {}x{}, grid {:?}, view bins {}, light bins {}",a.config.display(),c.render.width,c.render.height,c.cache.position_grid,c.cache.view_bins,c.cache.light_bins);
    println!("cache entries {} ; cache bytes {} ({:.1} MiB) ; working set estimate {:.2} GiB",c.cache.entries()?,c.cache.bytes()?,c.cache.bytes()? as f64/1048576.,c.working_set_bytes()? as f64/1073741824.);
    Ok(())
}

fn bake(a:&CfgArgs,out:&Path)->anyhow::Result<()> {
    let c=load(a)?;let s=runner::prepare(&c,None,false)?;
    s.gpu.save_cache(out,&c.cache,&s.teacher)?;
    println!("baked {} entries in {:.1} ms; wrote {} ({:.1} MiB)",c.cache.entries()?,s.bake_ms,out.display(),std::fs::metadata(out)?.len() as f64/1048576.);
    Ok(())
}

fn bench(a:BenchArgs)->anyhow::Result<()> {
    if a.jit_disk_cache{cutile::jit_cache::enable(std::sync::Arc::new(cutile::jit_cache::FileSystemJitStore::default_location()?));}
    let sequences:Vec<Sequence>=if a.sequence.iter().any(|s|s=="all"){vec![Sequence::Static,Sequence::Camera,Sequence::CameraLight]}else{a.sequence.iter().map(|s|s.parse()).collect::<anyhow::Result<_>>()?};
    let env=runner::environment()?;
    let mut report=Report{environment:env.clone(),notes:runner::notes(),records:Vec::new(),runs:Vec::new()};
    let mut text=String::new();
    let overridden=a.grid.is_some()||a.view_bins.is_some()||a.light_bins.is_some();
    for path in &a.configs {
        let mut base=Config::load(path)?;apply(&mut base,a.grid,a.view_bins,a.light_bins);
        if let Some(f)=a.frames{base.render.frames=f;base.benchmark.measured_frames=f;}
        if let Some(w)=a.warmup{base.benchmark.warmup_frames=w;}
        let resolutions=if a.resolution.is_empty(){vec![(base.render.width,base.render.height)]}else{a.resolution.iter().map(|r|parse_resolution(r)).collect::<anyhow::Result<_>>()?};
        let name=config_name(path,&base,overridden);
        for (w,h) in resolutions {
            let mut c=base.clone();c.render.width=w;c.render.height=h;
            let mut session=runner::prepare(&c,a.cache.as_deref(),a.modes.contains(&Mode::CacheInterp))?;
            for &seq in &sequences {
                let mut cs=c.clone();cs.render.sequence=seq;
                let (records,run)=runner::run_sequence(&mut session,&cs,&name,&a.modes,a.latency_frames.min(cs.render.frames),&env)?;
                let s=runner::summary(&env,&records);print!("{s}");text.push_str(&s);
                report.records.extend(records);report.runs.push(run);
                runner::write_report(&a.out,&report)?;std::fs::write(a.out.join("summary.txt"),&text)?;
            }
        }
    }
    println!("\nwrote {}/benchmark.json, benchmark.csv, summary.txt",a.out.display());
    Ok(())
}

/// Max abs RGB error per pixel, black-red-yellow-white ramp.
fn heatmap(a:&[f32],b:&[f32],scale:f32)->Vec<u8>{
    a.chunks_exact(3).zip(b.chunks_exact(3)).flat_map(|(x,y)|{
        let e=x.iter().zip(y).map(|(p,q)|(p-q).abs()).fold(0f32,f32::max);let t=(e/scale).clamp(0.,1.)*3.;
        [(t.min(1.)*255.) as u8,((t-1.).clamp(0.,1.)*255.) as u8,((t-2.).clamp(0.,1.)*255.) as u8]
    }).collect()
}
fn png(path:&Path,rgb:&[f32],w:usize,h:usize)->anyhow::Result<()>{let d:Vec<u8>=rgb.iter().map(|v|(v.clamp(0.,1.)*255.).round() as u8).collect();image::save_buffer(path,&d,w as u32,h as u32,image::ColorType::Rgb8)?;Ok(())}

fn render(a:&CfgArgs,seq:Sequence,frame:u32,out:&Path)->anyhow::Result<()> {
    let mut c=load(a)?;c.render.sequence=seq;
    let mut s=runner::prepare(&c,None,true)?;let g=&mut s.gpu;let f=frame as i32;
    g.reset(f)?;g.full(seq)?;let full=g.read_rgb()?;
    g.reset(f)?;g.cached(seq,&c.cache)?;let near=g.read_cache_rgb()?;
    g.reset(f)?;g.cached_interp(seq,&c.cache)?;let interp=g.read_cache_rgb()?;
    let (w,h)=(c.render.width,c.render.height);let scale=0.1;
    std::fs::create_dir_all(out)?;
    png(&out.join("full.png"),&full,w,h)?;png(&out.join("cache.png"),&near,w,h)?;png(&out.join("cache_interp.png"),&interp,w,h)?;
    image::save_buffer(out.join("error_heatmap.png"),&heatmap(&full,&near,scale),w as u32,h as u32,image::ColorType::Rgb8)?;
    image::save_buffer(out.join("error_heatmap_interp.png"),&heatmap(&full,&interp,scale),w as u32,h as u32,image::ColorType::Rgb8)?;
    let metrics=serde_json::json!({"resolution":format!("{w}x{h}"),"sequence":seq,"frame":frame,"cache":c.cache,
        "heatmap":"max abs RGB error per pixel, black=0, red=scale/3, yellow=2*scale/3, white>=scale","heatmap_scale":scale,
        "cache_nearest_vs_full":ErrorMetrics::compare(&full,&near),"cache_interp_vs_full":ErrorMetrics::compare(&full,&interp)});
    std::fs::write(out.join("metrics.json"),serde_json::to_vec_pretty(&metrics)?)?;
    println!("{}",serde_json::to_string_pretty(&metrics)?);Ok(())
}

const HOT_LOOP:&str=include_str!("../../narc-bench/src/hot_loop.rs");
const GPU_LIB:&str=include_str!("../../narc-gpu/src/lib.rs");
/// Gpu methods reachable from the measured loop.
const HOT_METHODS:[&str;10]=["reset","generate","teacher","full","lookup","cached","lookup_interp","cached_interp","replay","replay_full"];
const FORBIDDEN:[&str;9]=["to_host_vec","copy_host_vec_to_device","memcpy_htod","memcpy_dtoh","api::zeros","api::ones","api::dup","upload(","read_"];
fn body<'a>(src:&'a str,name:&str)->Option<&'a str>{
    let start=src.find(&format!("pub fn {name}("))?;let open=start+src[start..].find('{')?;let mut depth=0;
    for (i,ch) in src[open..].char_indices(){match ch{'{'=>depth+=1,'}'=>{depth-=1;if depth==0{return Some(&src[start..open+i+1]);}}_=>{}}}None
}
fn scan(hot_loop:&str,gpu_lib:&str)->anyhow::Result<Vec<String>> {
    let mut violations=Vec::new();
    let hot:Vec<&str>=hot_loop.lines().filter(|l|!l.trim_start().starts_with("//")).collect();
    for tok in FORBIDDEN{if hot.iter().any(|l|l.contains(tok)){violations.push(format!("hot_loop.rs contains `{tok}`"));}}
    for m in HOT_METHODS {
        let b=body(gpu_lib,m).ok_or_else(||anyhow::anyhow!("method {m} not found in narc-gpu"))?;
        for tok in FORBIDDEN{if b.contains(tok){violations.push(format!("Gpu::{m} contains `{tok}`"));}}
    }
    Ok(violations)
}
fn verify_zero_copy()->anyhow::Result<()> {
    println!("Intended residency (allocated once in Gpu::new / bake, never in the frame loop):");
    for (b,d) in [("frame counter","i32[1], advanced by a kernel; CUDA Graph replay reads it on device"),("features","f32[n,32] (17 used), written by generate_frame_features"),("teacher weights/biases","f32, uploaded once at Gpu::new"),("layer buffers","f32[n,128],[n,128],[n,64],[n,32], ping-pong"),("cache","f32[entries,4], baked on GPU (or loaded once from disk)"),("keys","i32[n]"),("rgb_cache","f32[n,4] output of every cache path")]{println!("  {b:<24} {d}");}
    let violations=scan(HOT_LOOP,GPU_LIB)?;
    println!("\nStatic scan of hot_loop.rs and Gpu::{{{}}} for {:?}:",HOT_METHODS.join(","),FORBIDDEN);
    if violations.is_empty(){println!("  no explicit host transfer or allocation in the measured path");}else{for v in &violations{println!("  VIOLATION {v}");}}
    println!("\nA static scan cannot see transfers issued inside libraries. Confirm with Nsight Systems:\n  scripts/verify-zero-copy.sh\nIt profiles only the measured interval (cuProfilerStart/Stop around hot_loop::measure) and counts memcpy HtoD/DtoH operations.");
    anyhow::ensure!(violations.is_empty(),"static zero-copy check failed");Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn measured_path_has_no_host_transfer(){assert_eq!(scan(HOT_LOOP,GPU_LIB).unwrap(),Vec::<String>::new());}
    #[test] fn scan_detects_injected_readback(){
        let injected=GPU_LIB.replacen("pub fn replay(&mut self)->anyhow::Result<()> {","pub fn replay(&mut self)->anyhow::Result<()> {let _=self.read_rgb();",1);
        assert_ne!(injected,GPU_LIB);assert!(!scan(HOT_LOOP,&injected).unwrap().is_empty());
    }
    #[test] fn narc_mode_requires_residual(){assert!("narc".parse::<Mode>().is_err());assert!("narc-graph".parse::<Mode>().is_err());assert_eq!("cache-only".parse::<Mode>().unwrap(),Mode::Cache);}
    #[test] fn resolution_parsing(){assert_eq!(parse_resolution("2560x1440").unwrap(),(2560,1440));assert!(parse_resolution("2560").is_err());}
}
