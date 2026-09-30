//! Measured frame loop. Kernel launches, graph launches and event records only:
//! no allocation, upload or readback between the profiler markers.
use crate::runner::Mode;
use cuda_core::Event;
use narc_core::config::Config;
use narc_gpu::Gpu;

pub struct Timings {pub pipelined:Vec<Vec<f64>>,pub latency:Vec<Vec<f64>>}
pub type Launch=fn(&mut Gpu,&Config)->anyhow::Result<()>;

pub fn launch(gpu:&mut Gpu,c:&Config,mode:Mode)->anyhow::Result<()> {
    match mode {
        Mode::Full=>gpu.full(c.render.sequence),
        Mode::FullGraph=>gpu.replay_full(),
        Mode::Cache=>gpu.cached(c.render.sequence,&c.cache),
        Mode::CacheInterp=>gpu.cached_interp(c.render.sequence,&c.cache),
        Mode::CacheGraph=>gpu.replay(),
    }
}

/// Throughput timing: batches enqueued without host waits, one event pair per frame.
/// Modes rotate at identical frame IDs to cancel clock drift.
fn pipelined(gpu:&mut Gpu,c:&Config,modes:&[Mode],frames:usize,events:&[(Event,Event)])->anyhow::Result<Vec<Vec<f64>>> {
    let batch=events.len()/modes.len();
    let mut order=Vec::with_capacity(events.len());
    let mut timings=vec![Vec::with_capacity(frames);modes.len()];
    let mut first=0;
    while first<frames {
        let count=batch.min(frames-first);order.clear();
        for frame in first..first+count {
            for j in 0..modes.len() {
                let m=(j+frame)%modes.len();let e=&events[order.len()];
                gpu.reset(frame as i32)?;
                e.0.record(&gpu.stream)?;launch(gpu,c,modes[m])?;e.1.record(&gpu.stream)?;
                order.push(m);
            }
        }
        events[order.len()-1].1.synchronize()?;
        for (i,&m) in order.iter().enumerate(){timings[m].push(events[i].0.elapsed_time(&events[i].1)? as f64);}
        first+=count;
    }
    Ok(timings)
}

/// Latency timing: host wait before each frame, so launch overhead is included.
fn latency(gpu:&mut Gpu,c:&Config,modes:&[Mode],frames:usize,(start,end):&(Event,Event))->anyhow::Result<Vec<Vec<f64>>> {
    let mut timings=vec![Vec::with_capacity(frames);modes.len()];
    for frame in 0..frames {
        for j in 0..modes.len() {
            let m=(j+frame)%modes.len();
            gpu.reset(frame as i32)?;gpu.finish()?;
            start.record(&gpu.stream)?;launch(gpu,c,modes[m])?;end.record(&gpu.stream)?;end.synchronize()?;
            timings[m].push(start.elapsed_time(&end)? as f64);
        }
    }
    Ok(timings)
}

pub fn measure(gpu:&mut Gpu,c:&Config,modes:&[Mode],latency_frames:usize)->anyhow::Result<Timings> {
    // Created outside the profiled interval.
    let pair=|g:&Gpu|->anyhow::Result<(Event,Event)>{Ok((g.device.new_event()?,g.device.new_event()?))};
    let events:Vec<(Event,Event)>=(0..32*modes.len()).map(|_|pair(gpu)).collect::<anyhow::Result<_>>()?;
    let single=pair(gpu)?;
    let jit_before=cutile::tile_kernel::jit_compile_count();
    gpu.finish()?;
    crate::profiler::start();
    let pipelined=pipelined(gpu,c,modes,c.render.frames,&events)?;
    let latency=latency(gpu,c,modes,latency_frames,&single)?;
    gpu.finish()?;
    crate::profiler::stop();
    anyhow::ensure!(jit_before==cutile::tile_kernel::jit_compile_count(),"JIT occurred during measurement: discard run");
    Ok(Timings{pipelined,latency})
}
