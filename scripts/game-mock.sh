#!/usr/bin/env bash
# Capture pipeline on the three deterministic mock sequences (CPU only).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
cargo build --release --bin narc
N="$CARGO_TARGET_DIR/release/narc"
: > out/game-mock-summary.txt
for k in static lighting dynamic; do
  "$N" game mock --kind "$k" --out "captures/mock/$k"
  "$N" game validate-capture "captures/mock/$k" | tee -a out/game-mock-summary.txt
  "$N" game run "captures/mock/$k" --every 8 2>/dev/null | tee -a out/game-mock-summary.txt
done
