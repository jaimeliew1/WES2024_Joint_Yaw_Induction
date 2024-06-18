from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from WES2024 import utils
from WES2024.Generate import diamond_AD
from WES2024.archive import LES_case_definitions

REGENERATE = False
FILESTEM = Path(__file__).stem


method_name_map = {
    "nocontrol": "NoControl",
    "jointcontrol": "JointControl",
    "yawcontrol": "YawControl",
    "thrustcontrol": "ThrustControl",
}


def generate(regenerate=False):
    df_LES = (
        LES_case_definitions.generate(regenerate=regenerate)
        .with_columns(type=pl.lit("cases"), method=pl.col("controller").replace(method_name_map))
        .select(pl.exclude("controller", "x", "y", "z"))
    )
    df_sweep = (
        diamond_AD.generate(regenerate=regenerate)
        .filter(pl.col("min_dist") == 6.0)
        .with_columns(
            type=pl.lit("sweep"),
        )
    )

    df_sweep = pl.concat([df_sweep, df_sweep.with_columns(-pl.col("wdir"))])
    df_sweep = df_sweep.select(df_LES.columns)
    df = pl.concat([df_LES, df_sweep])
    return df


def plot_wdir_sweep_abs(df: pl.DataFrame, ax: plt.Axes):
    # Plot sweep
    for method, _plot_params in utils.line_params.items():
        _df = df.filter(pl.col("type") == "sweep").filter(pl.col("method") == method)
        to_plot = _df.group_by("wdir").agg(pl.col("Cp").mean()).sort("wdir")
        ax.plot(to_plot["wdir"], to_plot["Cp"], **_plot_params)

    # Plot cases
    for method, _plot_params in utils.line_params.items():
        _df = df.filter(pl.col("type") == "cases").filter(pl.col("method") == method)
        to_plot = _df.group_by("wdir").agg(pl.col("Cp").mean()).sort("wdir")
        ax.plot(to_plot["wdir"], to_plot["Cp"], ".k")

    # Plot wind directions of interest at 6D
    for _wdir in df.filter(pl.col("type") == "cases")["wdir"].unique():
        ax.axvline(_wdir, lw=1, ls="--", c="k")

    ax.set_ylim(0.1, 0.6)

    ax.set_xlim(-20, 60)


def plot_wdir_sweep_rel(df: pl.DataFrame, ax: plt.Axes):
    # plot sweep
    for method, _plot_params in utils.line_params.items():
        _df = df.filter(pl.col("type") == "sweep").filter(pl.col("method") == method)
        to_plot = _df.group_by("wdir").agg(pl.col("Cp").mean()).sort("wdir")
        ref = (
            df.filter(pl.col("type") == "sweep")
            .filter(pl.col("method") == "NoControl")
            .group_by("wdir")
            .agg(pl.col("Cp").mean())
            .sort("wdir")
        )
        ax.plot(to_plot["wdir"], 100 * (to_plot["Cp"] / ref["Cp"] - 1), **_plot_params)

    # plot cases
    for method, _plot_params in utils.line_params.items():
        _df = df.filter(pl.col("type") == "cases").filter(pl.col("method") == method)
        to_plot = _df.group_by("wdir").agg(pl.col("Cp").mean()).sort("wdir")
        ref = (
            df.filter(pl.col("type") == "cases")
            .filter(pl.col("method") == "NoControl")
            .group_by("wdir")
            .agg(pl.col("Cp").mean())
            .sort("wdir")
        )
        ax.plot(to_plot["wdir"], 100 * (to_plot["Cp"] / ref["Cp"] - 1), ".k")

    # Plot wind directions of interest at 6D
    for _wdir in df.filter(pl.col("type") == "cases")["wdir"].unique():
        ax.axvline(_wdir, lw=1, ls="--", c="k")

    ax.set_xlim(-20, 60)


def plot_wdir_sweep(df: pl.DataFrame):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=np.array([8, 4]))
    plot_wdir_sweep_abs(df, axes[0])
    plot_wdir_sweep_rel(df, axes[1])

    # Legend
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4)

    # axes labels
    axes[1].set_xlabel("wind direction (deg)")
    axes[0].set_ylabel(r"$C_{P, \mathrm{farm}}$")
    axes[1].set_ylabel(r"$C_{P, \mathrm{farm}}$ increase (\%)")

    plt.savefig(utils.FIGDIRFORPAPER / f"{FILESTEM}_wdir_sweep.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=REGENERATE)
    plot_wdir_sweep(df)


if __name__ == "__main__":
    main()
