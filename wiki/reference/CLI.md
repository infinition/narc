# CLI

```text
narc <COMMAND>

  info              Device, toolchain and cache size estimates
  bake              Bake the dense cache on the GPU and save it
  bench             Measure FULL against the cache paths, write JSON/CSV
  render            Export images for one frame
  verify-zero-copy  Static scan of the measured path
  report            Markdown tables from benchmark.json files
  game              Offline pipeline on game captures
```

## Shared config options

`info`, `bake` and `render` accept:

| Option | Default | Meaning |
|---|---|---|
| `--config` | `configs/default.toml` | Base configuration |
| `--resolution` | from config | `WIDTHxHEIGHT` |
| `--grid` | from config | Cubic position grid |
| `--view-bins`, `--light-bins` | from config | Angular bins |

## bench

| Option | Default | Meaning |
|---|---|---|
| `--configs` | `configs/default.toml` | Comma-separated list |
| `--resolution` | from config | Comma-separated `WIDTHxHEIGHT` list |
| `--sequence` | `all` | `static`, `camera`, `camera-light`, `all` |
| `--modes` | `full,full-graph,cache,cache-interp,cache-graph` | Modes to measure |
| `--frames` | from config | Pipelined frames, also the quality frame count |
| `--latency-frames` | `100` | Latency-mode frames |
| `--warmup` | from config | Warmup frames |
| `--grid`, `--view-bins`, `--light-bins` | from config | Cache overrides |
| `--cache` | | Load a baked cache instead of baking |
| `--jit-disk-cache` | off | Persist cubins across processes |
| `--out` | `out/bench` | Output directory |

Modes:

| Mode | Path |
|---|---|
| `full` | Features + teacher MLP |
| `full-graph` | Same, CUDA Graph replay |
| `cache` | Nearest lookup |
| `cache-interp` | Multilinear lookup |
| `cache-graph` | Nearest lookup, CUDA Graph replay |

`narc` and `narc-graph` are rejected until the residual network exists.

## render

`--sequence` (default `camera`), `--frame` (default `120`), `--out` (default
`out/render`). Writes `full.png`, `cache.png`, `cache_interp.png`,
`error_heatmap.png`, `error_heatmap_interp.png`, `metrics.json`.

## game

| Command | Purpose |
|---|---|
| `mock --kind static\|lighting\|dynamic --out DIR` | Deterministic mock capture |
| `validate-capture CAPTURE [--pairs 5]` | Pre-bake checklist, `ready YES/NO` |
| `import CAPTURE` | Data level, cameras, depth range |
| `bake CAPTURE` | Cache from every Nth frame |
| `evaluate CAPTURE [--images 3]` | Reconstruct unseen frames, metrics and images |
| `run CAPTURE` | import + bake + evaluate |

Bake options: `--every 8`, `--voxel` (auto), `--voxel-scale 1.5`,
`--view-bins 8`, `--sky-bins 64`, `--no-normals`. Output defaults to
`out/game/<capture name>`.

## Environment

| Variable | Effect |
|---|---|
| `NARC_TEACHER_TILE=BMxBN` | Teacher output tile (default `8x128`) |
| `CUTILE_JIT_LOG=1` | cuTile JIT logging |
