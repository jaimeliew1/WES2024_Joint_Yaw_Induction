# R1 revision: data and figures

Run everything from the repository root with the project virtual environment:

```
python -m WES2024.R1.generate.regenerate_all          # data/ and figs/ from LES_DATA, ~3.5 h (workers: see the script)
python -m WES2024.R1.generate.regenerate_all --les    # first re-read LES_DATA/R1 from the raw LES (padeopsIO, /scratch)
python -m WES2024.R1.generate.regenerate_all --figures-only
```

## Layout

| Path | Contents |
|---|---|
| `generate/` | one script per dataset; `python -m WES2024.R1.generate.<script>` regenerates it |
| `forpaper/` | one script per figure; reads the stored data only and writes `figs/` |
| `wakemodels.py` | wake-model factory `get_windfarm` and the rotor constructors; TANDEM includes the ground by default |
| `controllers.py` | control strategies (`METHODS`), shared optimiser settings (`make_controller`, `VAR_SCALES`, `CT_MAX`), `use_annulus_optimum` |
| `farm_control_cases.py` | 5 x 5 farm layout; which LES run pairs with which set points |
| `les_data.py` | paths of the LES data in `LES_DATA/` |
| `LES_DATA/` (repository root) | reduced LES data, committed |
| `data/` (repository root) | model caches, git-ignored, regenerated from this folder; `data/extracted/` holds third-party digitized data that cannot be regenerated |

## Paper figures

| Figure | Script (`forpaper/`) | Model data (`generate/`) | LES data (`LES_DATA/`) |
|---|---|---|---|
| `plot_single_turbine_yaw_strategies` | same name | `generate_single_turbine_yaw_strategies` | |
| `plot_two_turbine_BEM_wdir_wakemodels_cp` | `plot_two_turbine_BEM_wakemodels_cp` | `generate_two_turbine_BEM_fine` | |
| `plot_two_turbine_BEM_wdir_varvortex` | `plot_two_turbine_BEM_wdir_fine` | `generate_two_turbine_BEM_fine` | |
| `twoturbine_les_contours` | same name | `generate_single_turbine_yaw_strategies` | `twoturbine_alldata_LES_std.csv` |
| `plot_power_yaw_twoturbine` | same name | | `twoturbine_alldata_LES_std.csv`, `R1/twoturbine_rews.csv`, `R1/uniform_inflow_les_delta_032.csv` |
| `twoturbine_gain_contours` | same name | `generate_two_turbine_contours` | `twoturbine_alldata_LES_std.csv` |
| `plot_les_tandem_hubplane` | same name | `generate_les_tandem_hubplane` | `R1/les_hubplane_Joint_TANDEM.npz`, `R1/abl_profiles.csv` |
| `plot_joint_setpoints` | same name (+ `setpoint_column`) | `generate_les_setpoints_tiled5x5` | |
| `plot_les_farm_gain` | same name (+ `print_les_farm_welch` for the statistics) | | `R1/les_farm_power_*.npz` |
| `plot_rowwise_power_umm`, `plot_rowwise_power_tabulated` | `plot_rowwise_power` | `generate_rowwise_power_models`, `generate_les_setpoints_tiled5x5` | `R1/les_farm_power_*.npz`, the two-turbine LES (rotor table) |

All BEM rotors read the look-up table of `generate_bem_lut_annulus`, and all rotors read the Unified Momentum Model tables of `generate_umm_lut`. `twoturbine_schematic` in the manuscript is a drawing, not made here.

Also made, not in the current manuscript: `plot_two_turbine_tamaro_gauss_gaussBP_vs_paper` (referee response; `generate_two_turbine_tamaro`, the digitized curves are read from
`data/extracted/tamaro_fig22/`, which is not committed; without them the script warns and saves `plot_two_turbine_tamaro_gauss_gaussBP` with only this study's data) and `plot_single_turbine_surface` (`generate_single_turbine_surface`).
