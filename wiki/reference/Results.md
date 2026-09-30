# Results

Generated with `narc report` from the JSON files in
[`results/`](https://github.com/infinition/narc/tree/main/results). Raw per-frame
timings and per-frame quality are in those files.

RTX 4070 Ti (sm_89), CUDA 13.3, Rust 1.98.1, cuTile Rust 0.3.1, Ubuntu 24.04 on
WSL2. 500 pipelined frames, 100 latency frames, quality over all 500 frame IDs.

## Headline matrix

Default cache: grid 24^3, 24 view bins, 12 light bins, 364.5 MB.

| config | resolution | sequence | mode | median GPU ms | p95 GPU ms | pipeline fps | speedup vs FULL | latency ms | PSNR dB | worst frame dB | RMSE | max abs err | hit rate | key reuse | cache MB |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 1280x720 | Static | FULL | 15.686 | 15.993 | 63.8 | 1.00 | 15.678 | ref | - | - | - | - | - | - |
| default | 1280x720 | Static | FULL_GRAPH | 15.673 | 15.970 | 63.8 | 1.00 | 15.663 | ref | - | - | - | - | - | - |
| default | 1280x720 | Static | CACHE_ONLY | 0.472 | 0.727 | 2118.4 | 33.23 | 0.486 | 33.15 | 33.15 | 0.02201 | 0.0906 | 100.0 | 100.0 | 364.5 |
| default | 1280x720 | Static | CACHE_INTERP | 0.769 | 0.988 | 1300.3 | 20.40 | 0.771 | 48.22 | 48.22 | 0.00388 | 0.0292 | 100.0 | 100.0 | 364.5 |
| default | 1280x720 | Static | CACHE_GRAPH | 0.461 | 0.708 | 2170.1 | 34.04 | 0.470 | 33.15 | 33.15 | 0.02201 | 0.0906 | 100.0 | 100.0 | 364.5 |
| default | 1280x720 | Camera | FULL | 15.731 | 16.017 | 63.6 | 1.00 | 15.715 | ref | - | - | - | - | - | - |
| default | 1280x720 | Camera | FULL_GRAPH | 15.728 | 16.006 | 63.6 | 1.00 | 15.706 | ref | - | - | - | - | - | - |
| default | 1280x720 | Camera | CACHE_ONLY | 0.472 | 0.708 | 2118.4 | 33.32 | 0.479 | 31.62 | 27.68 | 0.02625 | 0.1611 | 100.0 | 95.2 | 364.5 |
| default | 1280x720 | Camera | CACHE_INTERP | 0.761 | 0.993 | 1314.4 | 20.68 | 0.762 | 46.35 | 44.11 | 0.00482 | 0.0375 | 100.0 | 95.2 | 364.5 |
| default | 1280x720 | Camera | CACHE_GRAPH | 0.462 | 0.703 | 2165.3 | 34.06 | 0.471 | 31.62 | 27.68 | 0.02625 | 0.1611 | 100.0 | 95.2 | 364.5 |
| default | 1280x720 | CameraLight | FULL | 15.771 | 16.063 | 63.4 | 1.00 | 15.741 | ref | - | - | - | - | - | - |
| default | 1280x720 | CameraLight | FULL_GRAPH | 15.763 | 16.031 | 63.4 | 1.00 | 15.747 | ref | - | - | - | - | - | - |
| default | 1280x720 | CameraLight | CACHE_ONLY | 0.472 | 0.707 | 2118.4 | 33.41 | 0.496 | 35.84 | 30.95 | 0.01614 | 0.1324 | 100.0 | 94.8 | 364.5 |
| default | 1280x720 | CameraLight | CACHE_INTERP | 0.756 | 0.994 | 1323.3 | 20.87 | 0.761 | 48.67 | 44.88 | 0.00369 | 0.0440 | 100.0 | 94.8 | 364.5 |
| default | 1280x720 | CameraLight | CACHE_GRAPH | 0.462 | 0.710 | 2165.3 | 34.15 | 0.471 | 35.84 | 30.95 | 0.01614 | 0.1324 | 100.0 | 94.8 | 364.5 |
| default | 1920x1080 | Static | FULL | 35.895 | 36.612 | 27.9 | 1.00 | 36.046 | ref | - | - | - | - | - | - |
| default | 1920x1080 | Static | FULL_GRAPH | 35.949 | 36.530 | 27.8 | 1.00 | 36.133 | ref | - | - | - | - | - | - |
| default | 1920x1080 | Static | CACHE_ONLY | 1.052 | 1.370 | 950.9 | 34.13 | 1.062 | 33.15 | 33.15 | 0.02200 | 0.0905 | 100.0 | 100.0 | 364.5 |
| default | 1920x1080 | Static | CACHE_INTERP | 1.799 | 2.116 | 555.9 | 19.95 | 1.824 | 48.24 | 48.24 | 0.00387 | 0.0284 | 100.0 | 100.0 | 364.5 |
| default | 1920x1080 | Static | CACHE_GRAPH | 1.045 | 1.378 | 956.9 | 34.35 | 1.048 | 33.15 | 33.15 | 0.02200 | 0.0905 | 100.0 | 100.0 | 364.5 |
| default | 1920x1080 | Camera | FULL | 36.035 | 36.775 | 27.8 | 1.00 | 36.093 | ref | - | - | - | - | - | - |
| default | 1920x1080 | Camera | FULL_GRAPH | 36.055 | 36.725 | 27.7 | 1.00 | 36.150 | ref | - | - | - | - | - | - |
| default | 1920x1080 | Camera | CACHE_ONLY | 1.056 | 1.344 | 947.2 | 34.13 | 1.070 | 31.62 | 27.68 | 0.02625 | 0.1603 | 100.0 | 95.2 | 364.5 |
| default | 1920x1080 | Camera | CACHE_INTERP | 1.792 | 2.079 | 558.0 | 20.11 | 1.803 | 46.35 | 44.10 | 0.00481 | 0.0378 | 100.0 | 95.2 | 364.5 |
| default | 1920x1080 | Camera | CACHE_GRAPH | 1.047 | 1.345 | 955.5 | 34.43 | 1.046 | 31.62 | 27.68 | 0.02625 | 0.1603 | 100.0 | 95.2 | 364.5 |
| default | 1920x1080 | CameraLight | FULL | 37.414 | 38.841 | 26.7 | 1.00 | 36.228 | ref | - | - | - | - | - | - |
| default | 1920x1080 | CameraLight | FULL_GRAPH | 37.348 | 38.842 | 26.8 | 1.00 | 36.218 | ref | - | - | - | - | - | - |
| default | 1920x1080 | CameraLight | CACHE_ONLY | 1.087 | 1.559 | 919.6 | 34.40 | 1.102 | 35.84 | 30.95 | 0.01613 | 0.1381 | 100.0 | 94.8 | 364.5 |
| default | 1920x1080 | CameraLight | CACHE_INTERP | 1.807 | 2.279 | 553.4 | 20.71 | 1.825 | 48.67 | 44.88 | 0.00369 | 0.0445 | 100.0 | 94.8 | 364.5 |
| default | 1920x1080 | CameraLight | CACHE_GRAPH | 1.069 | 1.563 | 935.4 | 35.00 | 1.070 | 35.84 | 30.95 | 0.01613 | 0.1381 | 100.0 | 94.8 | 364.5 |
| default | 2560x1440 | Static | FULL | 65.324 | 68.204 | 15.3 | 1.00 | 65.362 | ref | - | - | - | - | - | - |
| default | 2560x1440 | Static | FULL_GRAPH | 65.337 | 68.256 | 15.3 | 1.00 | 65.406 | ref | - | - | - | - | - | - |
| default | 2560x1440 | Static | CACHE_ONLY | 1.927 | 2.240 | 518.9 | 33.90 | 1.879 | 33.15 | 33.15 | 0.02200 | 0.0908 | 100.0 | 100.0 | 364.5 |
| default | 2560x1440 | Static | CACHE_INTERP | 3.338 | 3.587 | 299.6 | 19.57 | 3.273 | 48.24 | 48.24 | 0.00387 | 0.0282 | 100.0 | 100.0 | 364.5 |
| default | 2560x1440 | Static | CACHE_GRAPH | 1.907 | 2.209 | 524.5 | 34.26 | 1.851 | 33.15 | 33.15 | 0.02200 | 0.0908 | 100.0 | 100.0 | 364.5 |
| default | 2560x1440 | Camera | FULL | 65.412 | 65.976 | 15.3 | 1.00 | 65.504 | ref | - | - | - | - | - | - |
| default | 2560x1440 | Camera | FULL_GRAPH | 65.452 | 65.993 | 15.3 | 1.00 | 65.633 | ref | - | - | - | - | - | - |
| default | 2560x1440 | Camera | CACHE_ONLY | 1.879 | 2.249 | 532.2 | 34.81 | 1.915 | 31.62 | 27.68 | 0.02625 | 0.1632 | 100.0 | 95.2 | 364.5 |
| default | 2560x1440 | Camera | CACHE_INTERP | 3.258 | 3.560 | 306.9 | 20.07 | 3.253 | 46.35 | 44.10 | 0.00482 | 0.0380 | 100.0 | 95.2 | 364.5 |
| default | 2560x1440 | Camera | CACHE_GRAPH | 1.872 | 2.239 | 534.2 | 34.94 | 1.898 | 31.62 | 27.68 | 0.02625 | 0.1632 | 100.0 | 95.2 | 364.5 |
| default | 2560x1440 | CameraLight | FULL | 65.717 | 69.875 | 15.2 | 1.00 | 66.041 | ref | - | - | - | - | - | - |
| default | 2560x1440 | CameraLight | FULL_GRAPH | 65.724 | 69.737 | 15.2 | 1.00 | 66.056 | ref | - | - | - | - | - | - |
| default | 2560x1440 | CameraLight | CACHE_ONLY | 1.931 | 2.495 | 517.8 | 34.03 | 1.949 | 35.84 | 30.95 | 0.01614 | 0.1380 | 100.0 | 94.8 | 364.5 |
| default | 2560x1440 | CameraLight | CACHE_INTERP | 3.298 | 3.718 | 303.2 | 19.93 | 3.262 | 48.67 | 44.88 | 0.00369 | 0.0448 | 100.0 | 94.8 | 364.5 |
| default | 2560x1440 | CameraLight | CACHE_GRAPH | 1.915 | 2.501 | 522.2 | 34.32 | 1.921 | 35.84 | 30.95 | 0.01614 | 0.1380 | 100.0 | 94.8 | 364.5 |

CUDA Graph, eager launches vs replay (median ms):

| config | resolution | sequence | path | eager pipelined | graph pipelined | eager latency | graph latency | graph == eager |
|---|---|---|---|---:|---:|---:|---:|---|
| default | 1280x720 | Static | FULL | 15.686 | 15.673 | 15.678 | 15.663 | true |
| default | 1280x720 | Camera | FULL | 15.731 | 15.728 | 15.715 | 15.706 | true |
| default | 1280x720 | CameraLight | FULL | 15.771 | 15.763 | 15.741 | 15.747 | true |
| default | 1920x1080 | Static | FULL | 35.895 | 35.949 | 36.046 | 36.133 | true |
| default | 1920x1080 | Camera | FULL | 36.035 | 36.055 | 36.093 | 36.150 | true |
| default | 1920x1080 | CameraLight | FULL | 37.414 | 37.348 | 36.228 | 36.218 | true |
| default | 2560x1440 | Static | FULL | 65.324 | 65.337 | 65.362 | 65.406 | true |
| default | 2560x1440 | Camera | FULL | 65.412 | 65.452 | 65.504 | 65.633 | true |
| default | 2560x1440 | CameraLight | FULL | 65.717 | 65.724 | 66.041 | 66.056 | true |
| default | 1280x720 | Static | CACHE nearest | 0.472 | 0.461 | 0.486 | 0.470 | true |
| default | 1280x720 | Camera | CACHE nearest | 0.472 | 0.462 | 0.479 | 0.471 | true |
| default | 1280x720 | CameraLight | CACHE nearest | 0.472 | 0.462 | 0.496 | 0.471 | true |
| default | 1920x1080 | Static | CACHE nearest | 1.052 | 1.045 | 1.062 | 1.048 | true |
| default | 1920x1080 | Camera | CACHE nearest | 1.056 | 1.047 | 1.070 | 1.046 | true |
| default | 1920x1080 | CameraLight | CACHE nearest | 1.087 | 1.069 | 1.102 | 1.070 | true |
| default | 2560x1440 | Static | CACHE nearest | 1.927 | 1.907 | 1.879 | 1.851 | true |
| default | 2560x1440 | Camera | CACHE nearest | 1.879 | 1.872 | 1.915 | 1.898 | true |
| default | 2560x1440 | CameraLight | CACHE nearest | 1.931 | 1.915 | 1.949 | 1.921 | true |

Speed x quality x memory, sorted by cache size (no ranking heuristic):

| cache MB | grid | view | light | resolution | sequence | lookup | median GPU ms | speedup vs FULL | PSNR dB | worst frame dB | max abs err | key reuse |
|---:|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 364.5 | 24x24x24 | 24 | 12 | 1280x720 | Camera | multilinear | 0.761 | 20.7 | 46.35 | 44.11 | 0.0375 | 95.2 |
| 364.5 | 24x24x24 | 24 | 12 | 1280x720 | Camera | nearest | 0.472 | 33.3 | 31.62 | 27.68 | 0.1611 | 95.2 |
| 364.5 | 24x24x24 | 24 | 12 | 1280x720 | CameraLight | multilinear | 0.756 | 20.9 | 48.67 | 44.88 | 0.0440 | 94.8 |
| 364.5 | 24x24x24 | 24 | 12 | 1280x720 | CameraLight | nearest | 0.472 | 33.4 | 35.84 | 30.95 | 0.1324 | 94.8 |
| 364.5 | 24x24x24 | 24 | 12 | 1280x720 | Static | multilinear | 0.769 | 20.4 | 48.22 | 48.22 | 0.0292 | 100.0 |
| 364.5 | 24x24x24 | 24 | 12 | 1280x720 | Static | nearest | 0.472 | 33.2 | 33.15 | 33.15 | 0.0906 | 100.0 |
| 364.5 | 24x24x24 | 24 | 12 | 1920x1080 | Camera | multilinear | 1.792 | 20.1 | 46.35 | 44.10 | 0.0378 | 95.2 |
| 364.5 | 24x24x24 | 24 | 12 | 1920x1080 | Camera | nearest | 1.056 | 34.1 | 31.62 | 27.68 | 0.1603 | 95.2 |
| 364.5 | 24x24x24 | 24 | 12 | 1920x1080 | CameraLight | multilinear | 1.807 | 20.7 | 48.67 | 44.88 | 0.0445 | 94.8 |
| 364.5 | 24x24x24 | 24 | 12 | 1920x1080 | CameraLight | nearest | 1.087 | 34.4 | 35.84 | 30.95 | 0.1381 | 94.8 |
| 364.5 | 24x24x24 | 24 | 12 | 1920x1080 | Static | multilinear | 1.799 | 20.0 | 48.24 | 48.24 | 0.0284 | 100.0 |
| 364.5 | 24x24x24 | 24 | 12 | 1920x1080 | Static | nearest | 1.052 | 34.1 | 33.15 | 33.15 | 0.0905 | 100.0 |
| 364.5 | 24x24x24 | 24 | 12 | 2560x1440 | Camera | multilinear | 3.258 | 20.1 | 46.35 | 44.10 | 0.0380 | 95.2 |
| 364.5 | 24x24x24 | 24 | 12 | 2560x1440 | Camera | nearest | 1.879 | 34.8 | 31.62 | 27.68 | 0.1632 | 95.2 |
| 364.5 | 24x24x24 | 24 | 12 | 2560x1440 | CameraLight | multilinear | 3.298 | 19.9 | 48.67 | 44.88 | 0.0448 | 94.8 |
| 364.5 | 24x24x24 | 24 | 12 | 2560x1440 | CameraLight | nearest | 1.931 | 34.0 | 35.84 | 30.95 | 0.1380 | 94.8 |
| 364.5 | 24x24x24 | 24 | 12 | 2560x1440 | Static | multilinear | 3.338 | 19.6 | 48.24 | 48.24 | 0.0282 | 100.0 |
| 364.5 | 24x24x24 | 24 | 12 | 2560x1440 | Static | nearest | 1.927 | 33.9 | 33.15 | 33.15 | 0.0908 | 100.0 |

## Cache size sweep

1920x1080, 300 frames, `camera` and `camera-light` sequences.

| cache MB | grid | view | light | resolution | sequence | lookup | median GPU ms | speedup vs FULL | PSNR dB | worst frame dB | max abs err | key reuse |
|---:|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 12.0 | 16x16x16 | 8 | 4 | 1920x1080 | Camera | multilinear | 1.791 | 20.1 | 30.63 | 29.03 | 0.1343 | 98.7 |
| 12.0 | 16x16x16 | 8 | 4 | 1920x1080 | Camera | nearest | 1.051 | 34.2 | 22.88 | 20.17 | 0.3390 | 98.7 |
| 12.0 | 16x16x16 | 8 | 4 | 1920x1080 | CameraLight | multilinear | 1.787 | 20.2 | 33.65 | 30.47 | 0.1284 | 98.7 |
| 12.0 | 16x16x16 | 8 | 4 | 1920x1080 | CameraLight | nearest | 1.055 | 34.3 | 27.66 | 22.82 | 0.3013 | 98.7 |
| 48.0 | 16x16x16 | 16 | 8 | 1920x1080 | Camera | multilinear | 1.798 | 20.0 | 39.75 | 37.73 | 0.0558 | 97.0 |
| 48.0 | 16x16x16 | 16 | 8 | 1920x1080 | Camera | nearest | 1.052 | 34.2 | 27.64 | 24.56 | 0.2305 | 97.0 |
| 48.0 | 16x16x16 | 16 | 8 | 1920x1080 | CameraLight | multilinear | 1.783 | 20.3 | 40.90 | 37.50 | 0.0574 | 96.7 |
| 48.0 | 16x16x16 | 16 | 8 | 1920x1080 | CameraLight | nearest | 1.055 | 34.4 | 31.37 | 26.86 | 0.1973 | 96.7 |
| 192.0 | 16x16x16 | 32 | 16 | 1920x1080 | Camera | multilinear | 1.790 | 20.1 | 46.92 | 45.52 | 0.0575 | 93.6 |
| 192.0 | 16x16x16 | 32 | 16 | 1920x1080 | Camera | nearest | 1.054 | 34.2 | 32.13 | 29.42 | 0.1707 | 93.6 |
| 192.0 | 16x16x16 | 32 | 16 | 1920x1080 | CameraLight | multilinear | 1.781 | 20.4 | 48.25 | 46.49 | 0.0630 | 93.0 |
| 192.0 | 16x16x16 | 32 | 16 | 1920x1080 | CameraLight | nearest | 1.055 | 34.4 | 35.30 | 31.93 | 0.1710 | 93.0 |
| 364.5 | 24x24x24 | 24 | 12 | 1920x1080 | Camera | multilinear | 1.797 | 20.1 | 46.16 | 44.10 | 0.0370 | 95.3 |
| 364.5 | 24x24x24 | 24 | 12 | 1920x1080 | Camera | nearest | 1.056 | 34.2 | 30.91 | 27.68 | 0.1603 | 95.3 |
| 364.5 | 24x24x24 | 24 | 12 | 1920x1080 | CameraLight | multilinear | 1.794 | 20.3 | 48.59 | 44.88 | 0.0445 | 95.0 |
| 364.5 | 24x24x24 | 24 | 12 | 1920x1080 | CameraLight | nearest | 1.056 | 34.4 | 35.99 | 30.95 | 0.1381 | 95.0 |
| 384.0 | 32x32x32 | 16 | 8 | 1920x1080 | Camera | multilinear | 1.810 | 19.9 | 40.47 | 38.28 | 0.0436 | 97.0 |
| 384.0 | 32x32x32 | 16 | 8 | 1920x1080 | Camera | nearest | 1.055 | 34.2 | 28.01 | 24.73 | 0.1873 | 97.0 |
| 384.0 | 32x32x32 | 16 | 8 | 1920x1080 | CameraLight | multilinear | 1.798 | 20.2 | 41.55 | 37.83 | 0.0422 | 96.7 |
| 384.0 | 32x32x32 | 16 | 8 | 1920x1080 | CameraLight | nearest | 1.058 | 34.3 | 32.30 | 27.18 | 0.1515 | 96.7 |
| 1536.0 | 32x32x32 | 32 | 16 | 1920x1080 | Camera | multilinear | 1.811 | 19.9 | 50.85 | 48.83 | 0.0283 | 93.6 |
| 1536.0 | 32x32x32 | 32 | 16 | 1920x1080 | Camera | nearest | 1.054 | 34.2 | 33.31 | 30.03 | 0.1217 | 93.6 |
| 1536.0 | 32x32x32 | 32 | 16 | 1920x1080 | CameraLight | multilinear | 1.796 | 20.2 | 52.73 | 49.32 | 0.0307 | 93.0 |
| 1536.0 | 32x32x32 | 32 | 16 | 1920x1080 | CameraLight | nearest | 1.057 | 34.3 | 38.17 | 33.21 | 0.1145 | 93.0 |
