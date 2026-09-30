# Capture Pipeline

`narc-game` reconstructs captured game frames offline. It answers a different
question from the synthetic benchmark: can a baked appearance representation
rebuild unseen views of a real scene, and does it beat plain depth reprojection?
A game does not run the NARC teacher, so a gain here is not a neural rendering
speedup.

## Methods

| Method | Description |
|---|---|
| `reprojection` | Backward warp from the two nearest keyframes, occlusion test against keyframe depth, bilinear sampling. Sky reprojected as directions. |
| `narc-nearest` | Sparse cache keyed by world voxel, octahedral view bin and normal class; containing voxel and bin. |
| `narc-interpolated` | 8 voxel centres x 4 view bins, empty corners skipped, voxel mean as fallback. |

## Protocol

- Keyframes are every Nth frame (`--every`). Summaries only cover the other
  frames; keyframe scores are a sanity check.
- Every method gets the same inputs: target depth and camera, keyframe color,
  depth and camera. No method sees the color of an unseen frame.
- Holes are counted (`coverage`), then filled with the nearest covered pixel on
  the row before full-frame metrics.
- Memory: reprojection keeps every keyframe (RGBA8 + D32), NARC keeps its cache.
- Metrics: PSNR, SSIM, RMSE, max error, coverage, temporal error
  (`(R_t - R_t-1) - (O_t - O_t-1)`), CPU time.

## Data levels

| Level | Data | Use |
|---|---|---|
| 0 | color | nothing geometric |
| 1 | + depth + camera | reprojection and NARC |
| 2 | + normals | NARC separates opposite faces |
| 3 | + motion vectors | planned: dynamic mask |
| 4 | + G-buffer, material, light | planned |

## Mock results

`scripts/game-mock.sh`: 640x360, 49 frames, 7 keyframes, 42 unseen frames.
CPU reference timings. Full output in
[`results/game-mock-summary.txt`](https://github.com/infinition/narc/blob/main/results/game-mock-summary.txt).

| Sequence | Method | PSNR dB | Worst dB | SSIM | Coverage | Temporal | Memory MB |
|---|---|---:|---:|---:|---:|---:|---:|
| static | reprojection | 39.97 | 37.20 | 0.9903 | 99.7 % | 0.01146 | 12.3 |
| static | narc-nearest | 29.31 | 28.29 | 0.9309 | 87.8 % | 0.03954 | 29.2 |
| static | narc-interpolated | 35.98 | 35.09 | 0.9804 | 97.0 % | 0.01878 | 29.2 |
| lighting | reprojection | 33.00 | 29.34 | 0.9723 | 99.7 % | 0.01855 | 12.3 |
| lighting | narc-nearest | 23.89 | 22.20 | 0.7603 | 87.8 % | 0.06668 | 29.2 |
| lighting | narc-interpolated | 26.81 | 23.92 | 0.9031 | 97.0 % | 0.03148 | 29.2 |
| dynamic | reprojection | 28.58 | 25.35 | 0.9707 | 98.4 % | 0.03064 | 12.3 |
| dynamic | narc-nearest | 24.72 | 22.99 | 0.8819 | 85.2 % | 0.05808 | 30.2 |
| dynamic | narc-interpolated | 27.08 | 24.27 | 0.9525 | 95.1 % | 0.03629 | 30.2 |

Depth reprojection beats the current cache on every mock sequence with less
memory: it keeps texture detail that voxel averaging loses. The cache has no
light dimension, so it averages lighting states. The mock validates the
pipeline, not the method; per-region metrics (specular, disocclusion) on real
captures are the next measurement.

## Next steps

1. First real capture, `validate-capture`, then a 40-frame static sequence.
2. Per-region metrics: specular, disocclusion, sky.
3. Dynamic mask from temporal difference, depth discontinuities, then motion
   vectors.
4. GPU ports of reprojection and cache lookup.
5. D3D12 to CUDA interop through shared resources and fences.
