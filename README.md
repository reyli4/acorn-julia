# ACORN: A Climate-informed Operational Reliability Network

This repo contains a Julia implementation of the New York State power system model developed by the [Anderson Research Group](https://andersonenergylab-cornell.github.io/). We implement the 2040 version of the model, originally detailed in [Lui et al. (2024)](https://arxiv.org/abs/2307.15079), which assumes a carbon-free grid based on the targets set in New York's Climate Leadership & Community Protection Act (CLCPA). The 2040 version is in turn based on a 2019 version which has been validated against current conditions, outlined in [Liu et al. (2023)](https://ieeexplore.ieee.org/document/9866561).

## Model info
- The grid infrastructure is described by a `branch`, `bus`, and `gen` file, located in `data/grid`.