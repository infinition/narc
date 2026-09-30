#!/usr/bin/env bash
# Headline benchmark: default cache, three resolutions, three sequences, all modes.
# One process per resolution, so cold JIT time and the VRAM delta are per resolution.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
cargo build --release --bin narc
BIN="$CARGO_TARGET_DIR/release/narc"
mkdir -p out/bench-main
: > out/bench-main.log
for res in 1280x720 1920x1080 2560x1440; do
  "$BIN" bench --configs configs/default.toml --resolution "$res" --sequence all \
    --modes full,full-graph,cache,cache-interp,cache-graph \
    --frames 500 --latency-frames 100 --warmup 50 --out "out/bench-main/$res" 2>&1 | tee -a out/bench-main.log
done
"$BIN" report out/bench-main/*/benchmark.json > out/RESULTS-main.md
