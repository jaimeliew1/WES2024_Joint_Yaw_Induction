"""
R1: regenerate everything behind the paper figures, in dependency order, then make the figures. Each step is one `python -m` of the script that
owns its data. Roughly 3.5 h if run in sequence: 25 min for the BEM look-up table (96 workers), 8 min for the farm optimiser (24 workers), 2.7 h for the
TANDEM sweep of generate_two_turbine_BEM_fine (12 workers), minutes for the rest.

    les      LES_DATA/R1 from the raw LES (needs padeopsIO and /scratch): only with --les; the committed files are used otherwise
    model    data/: Unified Momentum and BEM look-up tables, optimisers and model sweeps
    figures  R1/figs/

Usage: python -m WES2024.R1.generate.regenerate_all [--les] [--only NAME [NAME ...]] [--figures-only]
       NAME matches the step names below, e.g. --only setpoints rowwise
"""

import argparse
import os
import subprocess
import sys
import time

GEN = "WES2024.R1.generate."
FIG = "WES2024.R1.forpaper."

LES_STEPS = {
    "les_abl": [GEN + "generate_abl_profiles"],
    "les_farm_power": [GEN + "generate_les_farm_power"],
    "les_hubplane": [GEN + "generate_les_tandem_hubplane", "--les"],
}
MODEL_STEPS = {
    "umm_lut": [GEN + "generate_umm_lut"],
    "lut":[GEN + "generate_bem_lut_annulus"],
    "yaw_strategies": [GEN + "generate_single_turbine_yaw_strategies"],
    "surface": [GEN + "generate_single_turbine_surface"],
    "contours": [GEN + "generate_two_turbine_contours"],
    "wdir_sweep": [GEN + "generate_two_turbine_BEM_fine"],
    "tamaro": [GEN + "generate_two_turbine_tamaro"],
    "setpoints": [GEN + "generate_les_setpoints_tiled5x5", "--regenerate"],
    "rowwise": [GEN + "generate_rowwise_power_models"],
    "hubplane": [GEN + "generate_les_tandem_hubplane"],
}
FIGURE_STEPS = {
    "fig_yaw_strategies": [FIG + "plot_single_turbine_yaw_strategies"],
    "fig_surface": [FIG + "plot_single_turbine_surface"],
    "fig_wakemodels_cp": [FIG + "plot_two_turbine_BEM_wakemodels_cp"],
    "fig_wdir": [FIG + "plot_two_turbine_BEM_wdir_fine"],
    "fig_les_contours": [FIG + "twoturbine_les_contours"],
    "fig_power_yaw": [FIG + "plot_power_yaw_twoturbine"],
    "fig_gain_contours": [FIG + "twoturbine_gain_contours"],
    "fig_hubplane": [FIG + "plot_les_tandem_hubplane"],
    "fig_setpoints": [FIG + "plot_joint_setpoints"],
    "fig_les_gain": [FIG + "plot_les_farm_gain"],
    "fig_rowwise": [FIG + "plot_rowwise_power"],
    "fig_tamaro": [FIG + "plot_two_turbine_tamaro_vs_paper"],
    "print_welch": [FIG + "print_les_farm_welch"],
}


def run(name, command):
    print(f"\n=== {name}: {' '.join(command)}", flush=True)
    t0 = time.time()
    env = dict(os.environ, POLARS_MAX_THREADS="1", RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1")
    subprocess.run([sys.executable, "-m", *command], check=True, env=env)
    print(f"=== {name}: done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--les", action="store_true", help="first re-read LES_DATA/R1 from the raw LES")
    ap.add_argument("--only", nargs="+", default=None, help="run just these steps")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()
    steps = {} if args.figures_only else {**(LES_STEPS if args.les else {}), **MODEL_STEPS}
    steps.update(FIGURE_STEPS)
    if args.only:
        every = {**LES_STEPS, **MODEL_STEPS, **FIGURE_STEPS}
        steps = {name: every[name] for name in args.only}
    for name, command in steps.items():
        run(name, command)
