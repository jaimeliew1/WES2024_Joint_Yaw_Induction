"""
Figure 7:   
Diamond layout wind direction sweep for many layouts

This figure shows the optimal farm power output for the different control
strategies and for different turbine spacings.

Key points:
- ?
"""

from itertools import product
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from foreach import foreach
from mitwindfarm import Plotting
from mitwindfarm.Layout import Square
from mitwindfarm.windfarm import Windfarm

from optimise import JointControl, NoControl, ThrustControl, YawControl
from WES2024 import utils

FILESTEM = Path(__file__).stem

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

layouts = {
    4: Square(4.0, 5).rotate(45),
    5: Square(5.0, 5).rotate(45),
    6: Square(6.0, 5).rotate(45),
    7: Square(7.0, 5).rotate(45),
    8: Square(8.0, 5).rotate(45),
    9: Square(9.0, 5).rotate(45),
    10: Square(10.0, 5).rotate(45),
}
wdirs = np.arange(0, 360, 1)


# wdirs_of_interest = [0.0, 5.0, 26.565, 42.0, 45.0]
# @profile(filename="prof.prof")
def _generate(x):
    method, wdir, min_dist = x
    sol = methods[method](layouts[min_dist].rotate(wdir), windfarm).optimise(use_gradients=True)
    return utils.to_polars(sol).with_columns(
        pl.lit(method).alias("method"),
        pl.lit(wdir).alias("wdir"),
        pl.lit(min_dist).alias("min_dist"),
    )


@utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False):
    params = list(product(methods, wdirs, layouts))
    random.shuffle(params)

    df = pl.concat(foreach(_generate, params, parallel=True))
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
        for method, _plot_params in plot_params.items():
            _df = df.filter(pl.col("min_dist") == min_dist).filter(pl.col("method") == method)
            to_plot = _df.group_by("wdir").agg(pl.col("Cp").mean()).sort("wdir")
            ax.plot(to_plot["wdir"], to_plot["Cp"], **plot_params[method])

    axes[-1].set_xlabel("wind direction (deg)")
    [ax.set_ylabel("$C_P$") for ax in axes]

    # Plot wind directions of interest at 6D
    # for _wdir in wdirs_of_interest:
    #     axes[1].axvline(90 + _wdir, lw=1, ls="--", c="k")

    axes[0].set_ylim(0.1, 0.6)

    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4)

    axes[0].set_xlim(0, 360)
    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


# def plot_windfarm(df: pl.DataFrame):
#     for min_dist in df["min_dist"].unique():
#         _df = (
#             df.filter(pl.col("method") == "JointControl")
#             .filter(pl.col("min_dist") == min_dist)
#             .filter(pl.col("wdir") == 0)
#         )
#         windfarm_sol = utils.from_polars(_df, windfarm)

#         Plotting.plot_windfarm(windfarm_sol)
#         plt.savefig(utils.FIGDIR / f"square_windfarm_{min_dist}D.png", dpi=300, bbox_inches="tight")
#         plt.close()


# def plot_setpoints(df: pl.DataFrame):
#     df = df.filter(pl.col("min_dist") == 4).filter(pl.col("method") == "JointControl")

#     fig, axes = plt.subplots(2, 1, sharex=True)

#     for i, _df in df.group_by("turbine"):
#         axes[0].plot(_df["wdir"], np.rad2deg(_df["yaw"]), c=plt.cm.gist_ncar(i / 20))
#         axes[1].plot(_df["wdir"], _df["Ctprime"], c=plt.cm.gist_ncar(i / 20))

#     axes[-1].set_xlabel("wind direction [deg]")

#     axes[0].set_ylabel("yaw setpoint [deg]")
#     axes[1].set_ylabel("$C_T'$ setpoint")

#     axes[0].set_xlim(0, 360)
#     plt.savefig(utils.FIGDIR / "square_setpoints.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=False)
    # plot_setpoints(df)
    # plot_windfarm(df)
    # plot_farm_performance_vs_distance(df)
    plot_Cp_vs_distance(df)


if __name__ == "__main__":
    main()
