#!/usr/bin/env bash
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.3/bin:$PATH"
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-13.3/lib64:${LD_LIBRARY_PATH:-}"
# Build on the Linux filesystem: much faster than compiling on /mnt/c.
export CARGO_TARGET_DIR=/var/tmp/narc-target
