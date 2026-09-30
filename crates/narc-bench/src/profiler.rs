//! Nsight Systems capture range (`--capture-range=cudaProfilerApi`).
//! Symbols resolved from libcuda at run time; no-op without a profiler.
use std::sync::OnceLock;
type ProfilerFn=unsafe extern "C" fn()->u32;
struct Markers {_lib:libloading::Library,start:ProfilerFn,stop:ProfilerFn}
fn markers()->Option<&'static Markers>{
    static M:OnceLock<Option<Markers>>=OnceLock::new();
    M.get_or_init(||unsafe{
        let lib=["libcuda.so.1","libcuda.so"].iter().find_map(|n|libloading::Library::new(n).ok())?;
        let start=*lib.get::<ProfilerFn>(b"cuProfilerStart\0").ok()?;let stop=*lib.get::<ProfilerFn>(b"cuProfilerStop\0").ok()?;
        Some(Markers{_lib:lib,start,stop})
    }).as_ref()
}
pub fn start(){if let Some(m)=markers(){unsafe{(m.start)();}}}
pub fn stop(){if let Some(m)=markers(){unsafe{(m.stop)();}}}
