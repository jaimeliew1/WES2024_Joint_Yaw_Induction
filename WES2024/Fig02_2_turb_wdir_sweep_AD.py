from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from foreach import foreach
from mitwindfarm.Layout import Layout
from mitwindfarm.windfarm import Windfarm
from optimise import JointControl, NoControl, ThrustControl, YawControl

from WES2024 import utils

FILESTEM = Path(__file__).stem

windfarm = Windfarm()
layout = Layout([0, 7], [0.0, 0.0])
wdirs = np.arange(-20, 20, 0.25)


methods = {
    "NoControl": NoControl,
    "YawControl": YawControl,
    "ThrustControl": ThrustControl,
    "JointControl": JointControl,
}


def _generate(x):
    method, wdir = x
    sol = methods[method](layout.rotate(wdir), windfarm).optimise(Cp_constraint=None)

    return utils.to_polars(sol).with_columns(pl.lit(method).alias("method"), pl.lit(wdir).alias("wdir"))


@utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False):
    params = list(product(methods, wdirs))

    df = pl.concat(foreach(_generate, params, parallel=True))
    return df


def plot(df):
    fig, axes = plt.subplots(3, 1, sharex=True)

    for method in methods:
        _df = (
            df.filter(pl.col("method") == method)
            .group_by("wdir")
            .agg(pl.col("Cp").mean(), pl.col("yaw").first(), pl.col("Ctprime").first())
        ).sort("wdir")

        axes[0].plot(_df["wdir"], _df["Cp"], label=method)
        axes[1].plot(_df["wdir"], _df["Ctprime"], label=method)
        axes[2].plot(_df["wdir"], np.rad2deg(_df["yaw"]), label=method)

    axes[-1].set_xlabel("wind direction [deg]")

    axes[0].set_ylabel("$C_P$")
    axes[1].set_ylabel("$C_T'$")
    axes[2].set_ylabel(r"$\gamma$ [deg]")

    axes[0].set_ylim(0.3, 0.60)
    axes[0].legend()

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


# def plot_windfarms(df: pl.DataFrame):
#     for (method, wdir), _df in df.group_by(["method", "wdir"]):
#         windfarm_sol = utils.from_polars(_df, windfarm)
#         Plotting.plot_windfarm(windfarm_sol)
#         plt.savefig(utils.FIGDIR / f"{method}_{wdir}.png", dpi=300, bbox_inches="tight")
#         plt.close()


def main():
    df = generate(regenerate=False)
    plot(df)


if __name__ == "__main__":
    main()
