#!/usr/bin/env bash
# Zero-copy verification of the measured frame loop with Nsight Systems.
# The capture range is exactly hot_loop::measure (cuProfilerStart/Stop), so
# initialization uploads and post-run readbacks are outside the trace.
# Usage: scripts/verify-zero-copy.sh   (env: RES=1280x720 FRAMES=64 OUT=out/nsys)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
RES=${RES:-1280x720}; FRAMES=${FRAMES:-64}; OUT=${OUT:-out/nsys}
BIN="$CARGO_TARGET_DIR/release/narc"
mkdir -p "$OUT"
cargo build --release --bin narc
"$BIN" verify-zero-copy
nsys profile --force-overwrite=true --trace=cuda --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o "$OUT/hot-loop" "$BIN" bench --configs configs/default.toml --resolution "$RES" --sequence camera-light \
  --modes full,full-graph,cache,cache-interp,cache-graph --frames "$FRAMES" --latency-frames 16 --warmup 4 --out "$OUT/bench" > "$OUT/bench.log"
nsys stats --force-export=true --report cuda_api_sum,cuda_gpu_mem_size_sum,cuda_gpu_kern_sum --format csv \
  --output "$OUT/stats" "$OUT/hot-loop.nsys-rep" > "$OUT/stats.log" 2>&1 || true
api="$OUT/stats_cuda_api_sum.csv"
[ -s "$api" ] || { echo "no CUDA API trace in the capture range: verification inconclusive"; exit 2; }
# Sum "Num Calls" (column 3) of driver API rows whose name matches a pattern.
calls() { awk -F, -v re="$1" 'NR>1{n=$NF;gsub(/"/,"",n);c=$3;gsub(/"/,"",c);if(n~re)s+=c} END{print s+0}' "$api"; }
launches=$(calls '^cuLaunchKernel'); graphs=$(calls '^cuGraphLaunch')
htod=$(calls '^cuMemcpy.*HtoD'); dtoh=$(calls '^cuMemcpy.*DtoH'); dtod=$(calls '^cuMemcpy.*DtoD')
generic=$(calls '^cuMemcpy(Async)?(_v2)?$|^cuMemcpy(2D|3D|Peer|Batch)'); memset=$(calls '^cuMemset'); alloc=$(calls '^cuMem(Alloc|Free|HostAlloc|AllocHost)')
gpu_mem="$OUT/stats_cuda_gpu_mem_size_sum.csv"
gpu_note="GPU activity records unavailable (WSL2): verification at driver API level"
[ -s "$gpu_mem" ] && gpu_note="GPU activity records available, see $gpu_mem"
echo "measured interval: cuLaunchKernel=$launches cuGraphLaunch=$graphs"
echo "hot-path HtoD = $htod"
echo "hot-path DtoH = $dtoh"
echo "hot-path memcpy with unspecified direction = $generic ; DtoD = $dtod ; memset = $memset ; alloc/free = $alloc"
echo "$gpu_note"
printf '{"resolution":"%s","frames":%s,"cuLaunchKernel":%s,"cuGraphLaunch":%s,"htod_calls":%s,"dtoh_calls":%s,"memcpy_unspecified_calls":%s,"dtod_calls":%s,"memset_calls":%s,"alloc_free_calls":%s,"note":"%s"}\n' \
  "$RES" "$FRAMES" "$launches" "$graphs" "$htod" "$dtoh" "$generic" "$dtod" "$memset" "$alloc" "$gpu_note" > "$OUT/zero-copy.json"
[ $((launches+graphs)) -gt 0 ] || { echo "capture range is empty: verification inconclusive"; exit 2; }
if [ $((htod+dtoh+generic)) -eq 0 ]; then echo "ZERO-COPY VERIFIED for the measured interval"; else echo "ZERO-COPY VIOLATED"; exit 1; fi
