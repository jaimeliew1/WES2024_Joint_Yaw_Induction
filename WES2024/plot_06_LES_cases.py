from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from cache import cache_polars
from foreach import foreach
from mitwindfarm import Plotting
from mitwindfarm.Layout import Square
from mitwindfarm.windfarm import Windfarm
from optimise import JointControl, NoControl, ThrustControl, YawControl
from utilities import from_polars, to_polars


# Use Latex Fonts
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

pl.Config.set_tbl_rows(100)

FIGDIR = Path(__file__).parent.parent / "fig/LES_cases"
FIGDIR.mkdir(exist_ok=True, parents=True)

windfarm = Windfarm()


methods = {
    "NoControl": NoControl,
    "YawControl": YawControl,
    "ThrustControl": ThrustControl,
    "JointControl": JointControl,
}

plot_params = {
    "NoControl": dict(ls="--", c="k", label="No Control"),
    "ThrustControl": dict(c="tab:blue", label="Thrust Control"),
    "YawControl": dict(c="tab:orange", label="Yaw Control"),
    "JointControl": dict(c="tab:green", label="Joint Control"),
}

layout = Square(6.0, 5).rotate(45)
wdirs_of_interest = [-5.0, -2.5, 0.0, 2.5, 5.0]
wdirs_sweep = np.arange(-20.0, 20.0, 0.25)


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
    return to_polars(sol).with_columns(
        pl.lit(method).alias("method"),
        pl.lit(wdir).alias("wdir"),
    )


@cache_polars(Path(__file__).parent.parent / "data/plot_06_LES_LES_cases.csv")
def generate_LES_cases(regenerate=False):
    params = list(product(methods, wdirs_of_interest))

    df = pl.concat(foreach(_generate, params, parallel=True))
    return df


@cache_polars(Path(__file__).parent.parent / "data/plot_06_LES_wdir_sweep.csv")
def generate_wdir_sweep(regenerate=False):
    params = list(product(methods, wdirs_sweep))

    df = pl.concat(foreach(_generate, params, parallel=False))
    return df


@cache_polars(Path(__file__).parent.parent / "data/plot_06_LES.csv")
def generate(regenerate=False):
    df = pl.concat(
        [
            generate_LES_cases(regenerate).with_columns(type=pl.lit("cases")),
            generate_wdir_sweep(regenerate).with_columns(type=pl.lit("sweep"), wdir=pl.col("wdir").cast(float)),
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
    methods = ["NoControl", "ThrustControl", "YawControl", "JointControl"]
    df = df.filter(pl.col("type") == "cases")

    for wdir, _df in df.group_by("wdir"):
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=2 * np.array([4, 4]))

        for ax, method in zip(axes.ravel(), methods):
            windfarm_sol = from_polars(_df.filter(pl.col("method") == method), windfarm)
            Cp_ref = _df.filter(pl.col("method") == "NoControl")["Cp"].mean()

            Plotting.plot_windfarm(windfarm_sol, ax=ax)

            # plot turbine numbers
            for i, (x, y, _) in enumerate(windfarm_sol.layout):
                ax.text(x, y, f"{i+1}")

            ax.set_title(rf"{method} (Cp: {100*(windfarm_sol.Cp/Cp_ref - 1):+2.2f}\%)")
        plt.savefig(FIGDIR / f"LES_case_wdir{wdir:2.2f}.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_wdir_sweep(df: pl.DataFrame):

    plt.figure()
    ax = plt.gca()

    for method, _plot_params in plot_params.items():
        _df = df.filter(pl.col("type") == "sweep").filter(pl.col("method") == method)
        to_plot = _df.group_by("wdir").agg(pl.col("Cp").mean()).sort("wdir")
        ax.plot(to_plot["wdir"], to_plot["Cp"], **_plot_params)

    ax.set_xlabel("wind direction (deg)")
    ax.set_ylabel("$C_P$")

    # Plot wind directions of interest at 6D
    for _wdir in df.filter(pl.col("type") == "cases")["wdir"].unique():
        ax.axvline(_wdir, lw=1, ls="--", c="k")

    ax.set_ylim(0.1, 0.6)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4)

    ax.set_xlim(-20, 20)
    plt.savefig(FIGDIR / "wdir_sweep.png", dpi=300, bbox_inches="tight")


def plot_wdir_sweep_rel(df: pl.DataFrame):

    plt.figure()
    ax = plt.gca()

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

    ax.set_xlabel("wind direction (deg)")
    ax.set_ylabel(r"$C_P$ increase (\%)")

    # Plot wind directions of interest at 6D
    for _wdir in df.filter(pl.col("type") == "cases")["wdir"].unique():
        ax.axvline(_wdir, lw=1, ls="--", c="k")

    # ax.set_ylim(0.1, 0.6)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4)

    ax.set_xlim(-20, 20)
    plt.savefig(FIGDIR / "wdir_sweep_rel.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=False)
    plot_wdir_sweep(df)
    plot_wdir_sweep_rel(df)
    plot_windfarm(df)


if __name__ == "__main__":
    main()
