# NARC Wiki

**[infinition.github.io/narc](https://infinition.github.io/narc/)**

NARC (Neural Adaptive Rendering Cache) tests whether neural rendering can be
baked instead of run every frame. Passes such as NVIDIA DLSS 5 evaluate a large
network per pixel and per frame; NARC evaluates its network once, stores the
result in a GPU-resident cache and replaces per-frame inference with a lookup.

The proof of concept uses its own network so the full pass and the cache can be
compared exactly. It does not use or modify DLSS and is not affiliated with
NVIDIA.

## At a glance

| Item | Value |
|---|---|
| Teacher network | 17 -> 128 -> 128 -> 128 -> 64 -> 3 MLP, GELU, sigmoid, FP32 |
| Cache | Dense, keyed by object, 3D cell, view bin and light bin |
| Lookups | Nearest, multilinear (32 corners) |
| Reference GPU | RTX 4070 Ti, CUDA 13.3, cuTile Rust 0.3.1 |
| Cache vs full network | 33x to 35x (nearest), 19.6x to 20.9x (multilinear) |
| Multilinear quality | 46.4 to 48.7 dB PSNR against the full network |
| Hot path transfers | 0 HtoD, 0 DtoH |

## Getting started

- [Installation](Installation) - WSL2, CUDA toolkit, Rust toolchain
- [Quick Start](Quick-Start) - build, validate, benchmark, export images

## Concepts

- [Architecture](Architecture) - crates, kernels, data flow
- [Benchmark Methodology](Benchmark-Methodology) - timing, pairing, quality, JIT
- [Zero Copy](Zero-Copy) - definition and Nsight Systems verification
- [Limitations](Limitations) - what the dense cache does not solve

## Reference

- [CLI](CLI) - every `narc` command and option
- [Configuration](Configuration) - TOML fields
- [Results](Results) - full generated tables

## Captured game frames

- [Capture Pipeline](Capture-Pipeline) - offline reconstruction, reprojection baseline, first results
- [Capture Format](Capture-Format) - on-disk layout and metadata
- [Capture Guide](Capture-Guide) - recording a sequence with RenderDoc
