#!/usr/bin/env bash
set -euo pipefail
# Run as root inside Ubuntu 24.04 WSL. Never install a Linux NVIDIA driver in WSL.
apt-get update
apt-get install -y build-essential pkg-config libclang-dev curl ca-certificates git wget
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit-13-3
