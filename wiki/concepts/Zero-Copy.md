# Zero Copy

## Definition

Between the profiler markers around `hot_loop::measure`:

- no host-to-device or device-to-host copy;
- no device allocation or free;
- features are generated on the GPU from a device-side frame counter;
- weights, cache and intermediate buffers stay in VRAM.

Uploads at start-up and readbacks after measurement (quality statistics, images)
are outside the interval.

## Verification

1. `narc verify-zero-copy` scans `hot_loop.rs` and every `Gpu` method it reaches
   for host transfers or allocations. The same scan runs in `cargo test`.
2. `scripts/verify-zero-copy.sh` profiles the benchmark with Nsight Systems,
   `--capture-range=cudaProfilerApi`, so the trace covers exactly the measured
   interval, then counts driver API calls.

Under WSL2, Nsight Systems 2026.1.3 records the CUDA API trace but no GPU
activity. The check therefore works at the driver API level: any application
transfer between host and device is a `cuMemcpy*` call.

## Result

`results/zero-copy.json`, 1280x720, five modes:

```text
measured interval: cuLaunchKernel=1520 cuGraphLaunch=160
hot-path HtoD = 0
hot-path DtoH = 0
hot-path memcpy with unspecified direction = 0 ; DtoD = 0 ; memset = 0 ; alloc/free = 0
ZERO-COPY VERIFIED for the measured interval
```

Negative control (`results/zero-copy-negative-control.json`): one readback
injected into the interval is reported as `DtoH = 1`, `DtoD = 1`,
`alloc/free = 2`, and the check fails.
