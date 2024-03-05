from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from foreach import foreach
from mitwindfarm import Plotting
from mitwindfarm.windfarm import Windfarm
from optimise import JointControl, NoControl, ThrustControl, YawControl

from WES2024 import LES_case_definitions, utils

REGENERATE = False
FILESTEM = Path(__file__).stem


windfarm = Windfarm()


methods = {
    "nocontrol": NoControl,
    "jointcontrol": JointControl,
    "yawcontrol": YawControl,
    "thrustcontrol": ThrustControl,
}

plot_params = {
    "nocontrol": dict(ls="--", c="k", label="No Control"),
    "jointcontrol": dict(c="tab:green", label="Joint Control"),
    "yawcontrol": dict(c="tab:orange", label="Yaw Control"),
    "thrustcontrol": dict(c="tab:blue", label="Thrust Control"),
}

layout = LES_case_definitions.base_layout

wdirs_sweep = np.arange(-20.0, 60, 0.25)


#                                  _
#   __ _  ___ _ __   ___ _ __ __ _| |_ ___
#  / _` |/ _ \ '_ \ / _ \ '__/ _` | __/ _ \
# | (_| |  __/ | | |  __/ | | (_| | ||  __/
#  \__, |\___|_| |_|\___|_|  \__,_|\__\___|
#  |___/


# @profile(filename="prof.prof")
def _generate(x):
    method, wdir = x
    sol = methods[method](layout.rotate(wdir), windfarm).optimise(use_gradients=True)
    return utils.to_polars(sol).with_columns(
        pl.lit(method).alias("method"),
        pl.lit(wdir).alias("wdir"),
    )


@utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}_sweep.csv")
def generate_wdir_sweep(regenerate=False):
    params = list(product(methods, wdirs_sweep))

    df = pl.concat(foreach(_generate, params, parallel=False))
    return df


def generate(regenerate=False):
    df = pl.concat(
        [
            LES_case_definitions.generate(regenerate=regenerate)
            .rename({"controller": "method"})
            .with_columns(type=pl.lit("cases")),
            generate_wdir_sweep(regenerate=regenerate).with_columns(
                type=pl.lit("sweep"), wdir=pl.col("wdir").cast(float)
            ),
        ]
    )

    return df


#        _       _
#  _ __ | | ___ | |_
# | '_ \| |/ _ \| __|
# | |_) | | (_) | |_
# | .__/|_|\___/ \__|
# |_|


def plot_windfarm(df: pl.DataFrame):
    methods = ["nocontrol", "thrustcontrol", "yawcontrol", "jointcontrol"]
    df = df.filter(pl.col("type") == "cases")

    for wdir, _df in df.group_by("wdir"):
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=2 * np.array([4, 4]))

        for ax, method in zip(axes.ravel(), methods):
            windfarm_sol = utils.from_polars(_df.filter(pl.col("method") == method), windfarm)
            Cp_ref = _df.filter(pl.col("method") == "nocontrol")["Cp"].mean()

            Plotting.plot_windfarm(windfarm_sol, ax=ax)

            # plot turbine numbers
            for i, (x, y, _) in enumerate(windfarm_sol.layout):
                ax.text(x, y, f"{i+1}")

            ax.set_title(rf"{method} (Cp: {100*(windfarm_sol.Cp/Cp_ref - 1):+2.2f}\%)")
        plt.savefig(utils.FIGDIR / f"{FILESTEM}_{wdir:2.2f}.png", dpi=300, bbox_inches="tight")
        plt.close()


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
            .filter(pl.col("method") == "nocontrol")
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
            .filter(pl.col("method") == "nocontrol")
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

    # ax.set_ylim(0.1, 0.6)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4)

    ax.set_xlim(-20, 60)
    plt.savefig(utils.FIGDIR / f"{FILESTEM}_wdir_sweep_rel.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=REGENERATE)
    plot_wdir_sweep(df)
    plot_wdir_sweep_rel(df)
    plot_windfarm(df)


if __name__ == "__main__":
    main()
