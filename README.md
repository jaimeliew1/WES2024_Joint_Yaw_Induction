# WES2024: Joint Yaw-Induction Control for Wind Farms

Code, reduced LES data and figure scripts for the paper *Joint Yaw-Induction Control for Wind Farms* (Wind Energy Science).
The figures of the current revision are made by `WES2024/R1`; see [`WES2024/R1/README.md`](WES2024/R1/README.md) for which script
makes which figure.

## Installation

Requires Python 3.11-3.14 (tested with 3.12) and [Poetry](https://python-poetry.org/).

```bash
git clone https://github.com/jaimeliew1/WES2024_Joint_Yaw_Induction.git
cd WES2024_Joint_Yaw_Induction
poetry install
```

This installs the pinned versions of the wake and rotor models
([MITWindfarm](https://github.com/Howland-Lab/MITWindfarm), [MITRotor](https://github.com/Howland-Lab/MITRotor),
[Unified Momentum Model](https://github.com/Howland-Lab/Unified-Momentum-Model)) and the automatic-differentiation package
[Dualitic](https://github.com/jaimeliew1/Dualitic). Run the commands below inside the environment (`poetry shell`, or prefix
them with `poetry run`), from the repository root.

## Setup: look-up tables

The rotor models read look-up tables that are not in the repository. Build them once before anything else:

```bash
python -m WES2024.R1.generate.regenerate_all --only umm_lut lut
```

- `umm_lut`: Unified Momentum Model tables (`WES2024/unified_momentum_model_*_table.csv`), a few minutes.
- `lut`: BEM rotor table (`data/bem_lut_annulus_*.csv`), about 25 min on 96 workers; set the worker count with the `PROCESSES`
  environment variable (each worker needs about 0.2 GB).

## Reproducing the figures

```bash
python -m WES2024.R1.generate.regenerate_all                  # all model data in data/, then all figures (~3.5 h)
python -m WES2024.R1.generate.regenerate_all --figures-only   # figures only, from existing data/
python -m WES2024.R1.generate.regenerate_all --only NAME ...  # single steps, e.g. --only setpoints fig_setpoints
```

Figures are written to `WES2024/R1/figs/`. A high core-count machine is recommended (the slowest step, the TANDEM wind-direction
sweep, takes about 2.7 h on 12 workers).

## Data

| Path | Contents |
|---|---|
| `LES_DATA/` | reduced large-eddy simulation data (committed); the raw LES are not needed to make the figures |
| `data/` | model caches made by the scripts above (git-ignored) |
| `data/extracted/` | digitized curves from Tamaro et al. for one referee-response figure (not distributed; the script runs without them) |

`--les` re-reads `LES_DATA/R1` from the raw PadeOps output and only works where the raw LES are available.

## Repository layout

- `WES2024/R1/`: current revision: data generation (`generate/`), figures (`forpaper/`), wake models and controllers.
- `WES2024/` (top level): rotor models, gradients and optimisers shared with R1 (`CustomRotors.py`, `BEM_gradients.py`, `optimise.py`, ...).
- `WES2024/Generate`, `WES2024/Plot`, `scripts/`: the original submission; not maintained for the current dependency versions.
