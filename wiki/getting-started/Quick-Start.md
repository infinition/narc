# Quick Start

All commands run from the repository root after `source scripts/env.sh`.

## Build and test

```bash
cargo build --release
cargo test --release
```

## GPU validation

```bash
cargo run --release --example full_validate    # FULL vs independent CPU reference
cargo run --release --example cache_validate   # bake, keys, nearest and multilinear lookup vs CPU
cargo run --release --example graph_validate   # graph replay, invalidation, cache persistence
```

## Benchmark

```bash
cargo run --release --bin narc -- bench --resolution 1920x1080 --sequence camera-light
```

Writes `out/bench/benchmark.json`, `benchmark.csv` and `summary.txt`.

Reference runs:

```bash
bash scripts/bench-main.sh      # 720p, 1080p, 1440p x 3 sequences x 5 modes
bash scripts/bench-quality.sh   # six cache sizes at 1080p
```

## Images

```bash
cargo run --release --bin narc -- render --resolution 1920x1080 --sequence camera-light --frame 120 --out out/render
```

Exports the full network output, both cache lookups, error heatmaps and
`metrics.json`.

## Zero-copy check

```bash
bash scripts/verify-zero-copy.sh
```

## Capture pipeline on mock sequences

```bash
bash scripts/game-mock.sh
```
