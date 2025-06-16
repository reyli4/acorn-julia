# ACORN: A Climate-informed Operational Reliability Network

This repo contains a Julia implementation of the New York State power system model developed by the [Anderson Research Group](https://andersonenergylab-cornell.github.io/). For past work using ACORN, see [Liu et al. (2024)](https://arxiv.org/abs/2307.15079) and [Kabir et al. (2024)](https://doi.org/10.1016/j.renene.2024.120013). ACORN was validated using NYISO data from 2019, outlined in [Liu et al. (2023)](https://ieeexplore.ieee.org/document/9866561).

## Dependencies
This repo uses Python (mainly for data processing) and Julia (for running ACORN). Python dependencies are given in `pyproject.toml` and can be installed into a virtual environment using a tool like [uv](https://docs.astral.sh/uv/). Julia dependencies are given in `Project.toml`.

Paths to data/code are somewhat self-contained -- you will need to update the paths in `src/python/utils.py` and `src/julia/utils.jl`. If re-downloading all data from scratch using the scripts in `scripts/01_data_download`, you will need to update paths in all bash files. The initial experiments were performed on the Cornell Hopper cluster so some code may end up being specific to that system.

## Docs
See the following files for more information on the model construction and data processing: 
- [ACORN](docs/acorn.md)
- [Load modeling](docs/load_modeling.md)
- Solar + wind modeling
- [Coupling with GenX](docs/coupling_to_GenX.md) (including hydro, natural gas, nuclear matching)
- [Future work](docs/todo.md)