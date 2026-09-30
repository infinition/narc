#!/usr/bin/env bash
# Speed, quality and memory trade-off for a few cache sizes at 1920x1080.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
cargo build --release --bin narc
BIN="$CARGO_TARGET_DIR/release/narc"
common="--resolution 1920x1080 --sequence camera,camera-light --modes full,cache,cache-interp --frames 300 --latency-frames 30 --warmup 20"
mkdir -p out/bench-quality
: > out/bench-quality.log
run() { local name=$1; shift; "$BIN" bench "$@" $common --out "out/bench-quality/$name" 2>&1 | tee -a out/bench-quality.log; }
run speed        --configs configs/speed.toml
run g16-v16-l8   --configs configs/default.toml --grid 16 --view-bins 16 --light-bins 8
run g16-v32-l16  --configs configs/default.toml --grid 16 --view-bins 32 --light-bins 16
run default      --configs configs/default.toml
run g32-v16-l8   --configs configs/default.toml --grid 32 --view-bins 16 --light-bins 8
run quality      --configs configs/quality.toml
"$BIN" report out/bench-quality/*/benchmark.json > out/RESULTS-quality.md
