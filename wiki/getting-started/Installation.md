# Installation

NARC targets Linux. On Windows it runs under WSL2 with the Windows NVIDIA driver;
do not install a Linux NVIDIA driver inside WSL.

## Reference environment

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti (Ada, sm_89, 12 GB) |
| OS | Ubuntu 24.04 on WSL2, Windows 11 |
| CUDA toolkit | 13.3 |
| Rust | 1.98.1 (pinned in `rust-toolchain.toml`) |
| cuTile Rust | 0.3.1 (`cutile`, `cuda-core`) |
| Nsight Systems | 2026.1.3 (optional, zero-copy check) |

## Setup

As root in Ubuntu 24.04:

```bash
bash scripts/setup-wsl.sh
source scripts/env.sh
```

`setup-wsl.sh` installs build tools, rustup and `cuda-toolkit-13-3` from the
NVIDIA WSL repository. `env.sh` sets `PATH`, `CUDA_TOOLKIT_PATH`,
`LD_LIBRARY_PATH` and puts `CARGO_TARGET_DIR` on the Linux filesystem, which is
much faster than building on `/mnt/c`.

## Check

```bash
nvidia-smi
cargo run --release --example smoke
```

`smoke` runs the cuTile vector add example on the GPU.
