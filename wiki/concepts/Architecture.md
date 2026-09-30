# Architecture

## Data flow

```text
                 BAKE (once, GPU)
cell and bin centres -> teacher MLP -> dense cache (VRAM)
                                            |
                 RUNTIME (per frame, GPU)   |
frame counter -> features -> key ----------> lookup (nearest | multilinear) -> RGB
                     |
                     +-> teacher MLP -> RGB   (FULL path, reference)
```

Feature generation runs on the GPU from a device-side frame counter, so a
captured CUDA Graph renders a different frame on every replay without uploads.

## Crates

| Crate | Role |
|---|---|
| `narc-core` | Scene, teacher CPU reference, cache indexing, CPU interpolation reference, config, metrics, persistence |
| `narc-gpu` | cuTile kernels and `Gpu` state: persistent buffers, bake, lookups, CUDA Graphs, quality reductions |
| `narc-bench` | Measured loop, runner, JSON and CSV output, profiler capture markers |
| `narc-cli` | `narc` binary |
| `narc-game` | Offline reconstruction of captured game frames (CPU reference) |

## Scene

Six procedural material regions share a surface domain in [-1, 1]^3. The camera
orbits the z axis; the light follows a slower orbit. Three sequences:

| Sequence | Camera | Light |
|---|---|---|
| `static` | fixed | fixed |
| `camera` | 0.013 rad per frame | fixed |
| `camera-light` | 0.013 rad per frame | 0.0031 rad per frame |

Each sample carries 17 floats: position, normal, view and light directions,
albedo, roughness, metallic.

## Kernels

| Kernel | Purpose |
|---|---|
| `generate_frame_features` | Features for the current frame from the device frame counter |
| `teacher_layer` | One MLP layer, tiled `mma`, GELU or sigmoid |
| `generate_bake_features` | Features at cache cell and bin centres |
| `quantize_cache_key` | Dense key: object, x, y, z, view bin, light bin |
| `cache_lookup_nearest` | Gather, with per-object fallback to FULL |
| `cache_lookup_interp` | Multilinear over the five continuous axes |
| `error_rgb`, `reduce_mse` | Per-frame squared error and max error on the GPU |
| `key_match` | Temporal key reuse |
| `set_frame`, `advance` | Device-side frame counter |

## Teacher tile

The teacher output tile was swept once at 1080p
([results/teacher-tiles.log](https://github.com/infinition/narc/blob/main/results/teacher-tiles.log)).
8x128 is the default and is 2.25x faster than the initial 32x32 kernel. All
speedups are measured against this tuned FULL path. Override with
`NARC_TEACHER_TILE=BMxBN`.

## Persistence

Binary cache file with a JSON header: schema hash, teacher weight hash,
namespace versions (teacher, scene, material, quantization), quantization
parameters, per-object validity and a SHA-256 checksum. Incompatible caches are
rejected; invalidated objects fall back to the FULL path.
