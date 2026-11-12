# dualitic is imported first to ensure correct monkey patching.
import dualitic

"""
R1 paper figure: two-turbine BEM wake-model comparison, farm power coefficient only (6D aligned pair, all six control strategies, wind direction
sweep of generate/generate_two_turbine_BEM_fine.py), for the three wake models BP2016 Gaussian, vortex sheet and TANDEM (with the ground).

Usage: python -m WES2024.R1.forpaper.plot_two_turbine_BEM_wakemodels_cp
Saves R1/figs/plot_two_turbine_BEM_wdir_wakemodels_cp.png
"""

import matplotlib.pyplot as plt
import polars as pl

from WES2024.R1 import figuresettings as fs
from WES2024.R1.controllers import METHODS
from WES2024.R1.generate.generate_two_turbine_BEM_fine import load as load_fine
from WES2024.R1.wakemodels import DISPLAY_NAMES

MODELS = ["gaussBP", "varvortex", "tandem"]
UNBROKEN = ("NoControl", "ThrustControl")  # no yaw variable: continuous through wdir = 0


def load():
    dfs = [load_fine(model=model) for model in MODELS]
    return pl.concat(dfs, how="diagonal_relaxed").group_by("model", "method", "wdir").agg(pl.col("Cp").mean())


def plot(farm, stem):
    fig, axs = plt.subplots(1, len(MODELS), figsize=(7.0, 2.6), sharex=True, sharey=True)
    for ax, model in zip(axs, MODELS):
        for method in METHODS:
            d = farm.filter((pl.col("model") == model) & (pl.col("method") == method)).sort("wdir")
            # the sweeps have no wdir = 0 point for the yaw strategies; split there rather than draw a line across the (indeterminate) origin
            segments = [d] if method in UNBROKEN else [d.filter(pl.col("wdir") < 0), d.filter(pl.col("wdir") > 0)]
            for i, seg in enumerate(segments):
                ax.plot(seg["wdir"], seg["Cp"], c=fs.CONTROLLER_COLORS[method], ls=fs.CONTROLLER_LINESTYLES[method],
                        label=fs.controller_labels[method] if i == 0 else None)
        ax.set_title(f"({chr(97 + MODELS.index(model))}) {DISPLAY_NAMES[model]}", loc="left", fontsize=10)
        ax.set_xlabel("Wind direction (deg)")
    axs[0].set_ylabel(r"$C_{P, \mathrm{farm}}$")
    handles, labels = axs[0].get_legend_handles_labels()
    plt.subplots_adjust(wspace=0.12, bottom=0.34)  # leave room below the x labels for the legend
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=10, frameon=False)
    fs.save(stem)


if __name__ == "__main__":
    plot(load(), "plot_two_turbine_BEM_wdir_wakemodels_cp")
