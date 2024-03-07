from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from WES2024 import utils
from WES2024.Generate import LES_case_definitions, diamond_AD

REGENERATE = False
FILESTEM = Path(__file__).stem


method_name_map = {
    "nocontrol": "NoControl",
    "jointcontrol": "JointControl",
    "yawcontrol": "YawControl",
    "thrustcontrol": "ThrustControl",
}

plot_params = {
    "NoControl": dict(ls="--", c="k", label="No Control"),
    "JointControl": dict(c="tab:green", label="Joint Control"),
    "YawControl": dict(c="tab:orange", label="Yaw Control"),
    "ThrustControl": dict(c="tab:blue", label="Thrust Control"),
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


def plot_wdir_sweep(df: pl.DataFrame):
    plt.figure(figsize=(7, 3))
    ax = plt.gca()

    # Plot sweep
    for method, _plot_params in plot_params.items():
        _df = df.filter(pl.col("type") == "sweep").filter(pl.col("method") == method)
        to_plot = _df.group_by("wdir").agg(pl.col("Cp").mean()).sort("wdir")
        ax.plot(to_plot["wdir"], to_plot["Cp"], **_plot_params)

    # Plot cases
    for method, _plot_params in plot_params.items():
        _df = df.filter(pl.col("type") == "cases").filter(pl.col("method") == method)
        to_plot = _df.group_by("wdir").agg(pl.col("Cp").mean()).sort("wdir")
        ax.plot(to_plot["wdir"], to_plot["Cp"], ".k")

    ax.set_xlabel("wind direction (deg)")
    ax.set_ylabel("$C_P$")

    # Plot wind directions of interest at 6D
    for _wdir in df.filter(pl.col("type") == "cases")["wdir"].unique():
        ax.axvline(_wdir, lw=1, ls="--", c="k")

    ax.set_ylim(0.1, 0.6)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4)

    ax.set_xlim(-20, 60)
    plt.savefig(utils.FIGDIR / f"{FILESTEM}_wdir_sweep.png", dpi=300, bbox_inches="tight")


def plot_wdir_sweep_rel(df: pl.DataFrame):
    plt.figure(figsize=(7, 3))
    ax = plt.gca()

    # plot sweep
    for method, _plot_params in plot_params.items():
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
    for method, _plot_params in plot_params.items():
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

    ax.set_xlabel("wind direction (deg)")
    ax.set_ylabel(r"$C_P$ increase (\%)")

    # Plot wind directions of interest at 6D
    for _wdir in df.filter(pl.col("type") == "cases")["wdir"].unique():
        ax.axvline(_wdir, lw=1, ls="--", c="k")

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4)

    ax.set_xlim(-20, 60)
    plt.savefig(utils.FIGDIR / f"{FILESTEM}_wdir_sweep_rel.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=REGENERATE)
    plot_wdir_sweep(df)
    plot_wdir_sweep_rel(df)


if __name__ == "__main__":
    main()
