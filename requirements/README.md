# Dependency profiles

- `cuda121.txt`: NVIDIA/CUDA 12.1 runtime, recommended for current RTX servers.
- `cuda118.txt`: NVIDIA/CUDA 11.8 runtime.
- `cpu.txt`: CPU-only development and continuous integration.
- `base.txt`: exact shared experiment dependencies.
- `dev.txt`: tests and linting only.

The files under this directory pin direct dependencies. The root-level
`requirements*.txt` files additionally install SpecLoc itself in editable mode.
MMCV is restricted to a pre-built wheel so installation fails fast instead of
silently compiling from source.
