# Benchmark Methodology

## Timing

All frame times are CUDA event times. Wall clocks are only used for bake and
JIT.

| Metric | Protocol |
|---|---|
| `median_gpu_ms`, `p95_gpu_ms` | Pipelined: batches of 32 frames enqueued without host waits, one event pair around each frame pipeline. Device execution time. |
| `latency_median_ms` | One event pair per frame after a host wait. Includes launch overhead; the regime where CUDA Graphs can help. |
| `equivalent_fps` | `1000 / median_gpu_ms` of this pipeline. Not a game frame rate. |

Every mode renders the same frame IDs, interleaved in rotating order within the
same process, to cancel clock and thermal drift. Feature generation is included
in every mode.

## JIT

cuTile specializes kernels on the power-of-two divisibility of integer scalar
arguments (up to 16). Every frame value class and sequence code is compiled
before measurement. The process JIT counter is checked around the measured
interval and any JIT aborts the run. `cold_compile_ms` and `bake_ms` are
reported separately.

## Quality

For every measured frame ID, FULL runs once, then each cache path renders the
same frame. Squared error and max absolute error are reduced on the GPU and read
back once at the end. Reported: MSE, RMSE, PSNR (peak 1.0) over all frames, worst
frame PSNR and max absolute error.

## Hit rate and reuse

The dense cache covers the whole scene domain, so `cache_hit_rate` is 100 % by
construction: it only reports a valid namespace. Actual reuse under motion is
`temporal_key_reuse`, the fraction of samples whose nearest key is unchanged
since the previous frame.

## Memory

`allocated_vram_bytes` is the exact size of the persistent buffers.
`measured_vram_bytes` is the free-memory delta before allocation and before
measurement; `bench-main.sh` runs one process per resolution so that it stays
valid.

## Fairness

- FULL uses the fastest teacher tile found by a sweep, not the initial kernel.
- FULL is still a plain FP32 tiled MLP. A fused or TF32/FP16 teacher would reduce
  every speedup.
- There is no residual network yet, so there is no `NARC` mode. The CLI rejects
  `--modes narc`.
