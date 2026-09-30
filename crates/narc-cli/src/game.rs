//! `narc game`: offline pipeline on captured frames.
use clap::{Args,Subcommand};
use narc_game::{capture::FrameSource,mock::MockKind,narc::NarcConfig,pipeline,DiskCapture};
use std::path::{Path,PathBuf};

#[derive(Subcommand)]
pub enum GameCmd {
    /// Write a deterministic mock capture.
    Mock {#[arg(long,default_value="static")] kind:MockKind,#[arg(long,default_value_t=640)] width:usize,#[arg(long,default_value_t=360)] height:usize,#[arg(long,default_value_t=49)] frames:usize,#[arg(long)] out:PathBuf},
    /// Report data level, cameras and depth.
    Import {capture:PathBuf,#[arg(long)] out:Option<PathBuf>},
    /// Bake the cache from every Nth frame.
    Bake {capture:PathBuf,#[command(flatten)] bake:BakeArgs,#[arg(long)] out:Option<PathBuf>},
    /// Reconstruct unseen frames, write images and metrics.
    Evaluate {capture:PathBuf,#[arg(long)] bake:Option<PathBuf>,#[arg(long,default_value_t=3)] images:usize,#[arg(long)] out:Option<PathBuf>},
    /// Pre-bake checklist.
    ValidateCapture {capture:PathBuf,#[arg(long,default_value_t=5)] pairs:usize,#[arg(long)] out:Option<PathBuf>},
    /// import + bake + evaluate.
    Run {capture:PathBuf,#[command(flatten)] bake:BakeArgs,#[arg(long,default_value_t=3)] images:usize,#[arg(long)] out:Option<PathBuf>},
}

#[derive(Args,Clone)]
pub struct BakeArgs {
    /// Keyframe spacing.
    #[arg(long,default_value_t=8)] every:usize,
    /// World-space voxel size (auto when absent).
    #[arg(long)] voxel:Option<f64>,
    #[arg(long,default_value_t=1.5)] voxel_scale:f64,
    #[arg(long,default_value_t=8)] view_bins:u32,
    #[arg(long,default_value_t=64)] sky_bins:u32,
    /// Ignore normals even when present.
    #[arg(long)] no_normals:bool,
}
impl BakeArgs {fn config(&self)->NarcConfig{NarcConfig{voxel:self.voxel,voxel_scale:self.voxel_scale,view_bins:self.view_bins,sky_bins:self.sky_bins,use_normals:!self.no_normals}}}

fn out_dir(out:&Option<PathBuf>,cap:&DiskCapture)->PathBuf{out.clone().unwrap_or_else(||Path::new("out/game").join(&cap.manifest.name))}

pub fn run(cmd:GameCmd)->anyhow::Result<()> {
    match cmd {
        GameCmd::Mock{kind,width,height,frames,out}=>{let m=narc_game::mock::generate(&out,kind,width,height,frames)?;println!("wrote mock capture {} ({} frames, {}x{}) to {}",m.name,m.frames.len(),width,height,out.display());Ok(())}
        GameCmd::ValidateCapture{capture,pairs,out}=>{let c=DiskCapture::open(&capture)?;validate(&c,pairs,&out_dir(&out,&c))}
        GameCmd::Import{capture,out}=>{let c=DiskCapture::open(&capture)?;import(&c,&out_dir(&out,&c)).map(|_|())}
        GameCmd::Bake{capture,bake,out}=>{let c=DiskCapture::open(&capture)?;bake_step(&c,&capture,&bake,&out_dir(&out,&c)).map(|_|())}
        GameCmd::Evaluate{capture,bake,images,out}=>{let c=DiskCapture::open(&capture)?;let o=out_dir(&out,&c);evaluate(&c,&bake.unwrap_or_else(||o.clone()),&o,images)}
        GameCmd::Run{capture,bake,images,out}=>{
            let c=DiskCapture::open(&capture)?;let o=out_dir(&out,&c);
            let r=import(&c,&o)?;anyhow::ensure!(r.geometry_usable,"capture is not usable for reconstruction: {:?}",r.issues);
            bake_step(&c,&capture,&bake,&o)?;evaluate(&c,&o,&o,images)
        }
    }
}

fn import(c:&DiskCapture,out:&Path)->anyhow::Result<pipeline::ImportReport> {
    let r=pipeline::import(c)?;std::fs::create_dir_all(out)?;std::fs::write(out.join("import.json"),serde_json::to_vec_pretty(&r)?)?;
    println!("{} ({}): {} frames {} ; {} ; {:?} ; cameras {}/{} ; sky {:.1}% ; camera path {:.3} ; geometry usable: {} ; camera check frame 0->1: PSNR {} dB over {} covered",r.name,r.source,r.frames,r.resolution,r.level_description,r.scene_class,r.frames_with_camera,r.frames,r.sky_fraction*100.,r.camera_path_length,r.geometry_usable,r.pair_check_psnr_db.map_or("-".into(),|v|format!("{v:.1}")),r.pair_check_coverage.map_or("-".into(),|v|format!("{:.1}%",v*100.)));
    for i in &r.issues{println!("  issue: {i}");}
    Ok(r)
}
fn bake_step(c:&DiskCapture,capture:&Path,a:&BakeArgs,out:&Path)->anyhow::Result<pipeline::BakeReport> {
    let r=pipeline::bake(c,capture,out,a.every,&a.config())?;
    println!("baked {} keyframes (every {}) in {:.0} ms CPU: voxel {:.4}, normals in key {}, {} entries, cache {:.1} MB vs keyframe store {:.1} MB",r.keyframes.len(),r.every,r.bake_cpu_ms,r.voxel,r.normals_in_key,r.cache_entries,r.cache_bytes as f64/1048576.,r.keyframe_store_bytes as f64/1048576.);
    Ok(r)
}
fn evaluate(c:&DiskCapture,bake:&Path,out:&Path,images:usize)->anyhow::Result<()> {
    anyhow::ensure!(c.len()>1,"need at least two frames");
    let r=pipeline::evaluate(c,bake,out,images)?;
    let s=pipeline::print_summary(&r);print!("{s}");std::fs::write(out.join("summary.txt"),&s)?;
    println!("wrote {}/metrics.json, metrics.csv, summary.txt and frame_*/ images",out.display());
    Ok(())
}
fn validate(c:&DiskCapture,pairs:usize,out:&Path)->anyhow::Result<()> {
    let v=pipeline::validate(c,pairs)?;
    std::fs::create_dir_all(out)?;std::fs::write(out.join("validation.json"),serde_json::to_vec_pretty(&v)?)?;
    println!("{} ({}) - {}",v.import.name,v.import.source,v.import.level_description);
    for k in &v.checks{println!("{:<14} {:<4} {}",k.name,if k.ok{"OK"}else{"FAIL"},k.detail);}
    for i in &v.import.issues{println!("  issue: {i}");}
    println!("{:<14} {}","ready",if v.ready{"YES"}else{"NO"});
    anyhow::ensure!(v.ready,"capture is not ready for baking");Ok(())
}
