# Dependency profiles

- `cuda121.txt`: NVIDIA/CUDA 12.1 runtime, recommended for current RTX servers.
- `cuda118.txt`: NVIDIA/CUDA 11.8 runtime.
- `cpu.txt`: CPU-only development and continuous integration.
- `base.txt`: exact shared experiment dependencies.
- `dev.txt`: tests and linting only.
- `environment.yml`: optional Conda/CUDA 11.8 environment used by the cloud script.

The files under this directory pin direct dependencies. The root-level
`requirements.txt` is the single default CUDA 12.1 entry point and also installs
SpecLoc itself in editable mode. For another profile, install the selected file
and then run `python -m pip install -e .`. MMCV is restricted to a pre-built
wheel so installation fails fast instead of silently compiling from source.
