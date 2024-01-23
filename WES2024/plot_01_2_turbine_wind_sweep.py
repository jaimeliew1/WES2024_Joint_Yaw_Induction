from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from cache import cache_polars
from foreach import foreach
from mitwindfarm import Plotting
from mitwindfarm.Layout import Layout
from mitwindfarm.windfarm import Windfarm
from optimise import JointControl, NoControl, ThrustControl, YawControl

from utilities import to_polars, from_polars

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)

windfarm = Windfarm()
layout = Layout([0, 7], [0.0, 0.0])


methods = {
    "NoControl": NoControl,
    "YawControl": YawControl,
    "ThrustControl": ThrustControl,
    "JointControl": JointControl,
}


def _generate(x):
    method, wdir = x
    sol = methods[method](layout.rotate(wdir), windfarm).optimise()

    return to_polars(sol).with_columns(
        pl.lit(method).alias("method"), pl.lit(wdir).alias("wdir")
    )


@cache_polars(Path(__file__).parent.parent / "data/plot_01_2_turbine_wind_sweep.csv")
def generate(regenerate=False):
    wdirs = np.arange(-20, 20, 0.25)
    params = list(product(methods, wdirs))

    df = pl.concat(foreach(_generate, params, parallel=False))
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
    axes[2].set_ylabel("$\gamma$ [deg]")

    axes[0].set_ylim(0.3, 0.60)
    axes[0].legend()

    plt.savefig(FIGDIR / "wind_direction_sweep.png", dpi=300, bbox_inches="tight")


def plot_windfarms(df: pl.DataFrame):
    for (method, wdir), _df in df.group_by(["method", "wdir"]):
        windfarm_sol = from_polars(_df, windfarm)
        Plotting.plot_windfarm(windfarm_sol)
        plt.savefig(FIGDIR / f"{method}_{wdir}.png", dpi=300, bbox_inches="tight")
        plt.close()


def main():
    df = generate(regenerate=False)
    plot(df)
    # plot_windfarms(df)


if __name__ == "__main__":
    main()
