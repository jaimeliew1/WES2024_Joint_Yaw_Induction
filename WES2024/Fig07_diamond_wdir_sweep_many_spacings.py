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
from mitwindfarm.Layout import Square
from mitwindfarm.windfarm import Windfarm

from optimise import JointControl, NoControl, ThrustControl, YawControl
from WES2024 import utils

REGENERATE = False
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
wdirs = np.arange(0.0, 90.0, 0.05)


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

    axes[0].set_ylim(0.1, 0.6)

    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4)

    axes[0].set_xlim(0, 360)
    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=REGENERATE)
    plot_Cp_vs_distance(df)


if __name__ == "__main__":
    main()
