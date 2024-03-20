"""
Figure 7:   
Diamond layout wind direction sweep for many layouts

This figure shows the optimal farm power output for the different control
strategies and for different turbine spacings.

Key points:
- ?
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from WES2024.Generate import diamond_AD

from WES2024 import utils

FILESTEM = Path(__file__).stem
REGENERATE = False


def generate(regenerate=False):
    df_quarter = diamond_AD.generate(regenerate=regenerate)
    df = utils.fill_in_other_quadrants(df_quarter)
    return df


def plot_Cp_vs_distance(df: pl.DataFrame):
    N_dist = df["min_dist"].n_unique()

    fig, axes = plt.subplots(N_dist, 1, sharex=True, sharey=True)

    for ax, min_dist in zip(axes, df["min_dist"].unique().sort()):
        ax.text(
            0.5,
            0.99,
            f"turbine spacing: {min_dist}D",
            ha="center",
            va="top",
            transform=ax.transAxes,
        )
        for method, _plot_params in utils.line_params.items():
            _df = df.filter(pl.col("min_dist") == min_dist).filter(pl.col("method") == method)
            to_plot = _df.group_by("wdir").agg(pl.col("Cp").mean()).sort("wdir")
            ax.plot(to_plot["wdir"], to_plot["Cp"], **_plot_params)

    axes[-1].set_xlabel("wind direction (deg)")
    [ax.set_ylabel("$C_P$") for ax in axes]

    axes[0].set_ylim(0.1, 0.6)

    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4)

    axes[0].set_xlim(0, 360)
    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=REGENERATE)
    plot_Cp_vs_distance(df)


if __name__ == "__main__":
    main()
