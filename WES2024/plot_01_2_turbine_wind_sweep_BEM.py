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
from mitwindfarm.Rotor import BEM
from test_BEM_gradients import DualBEM
from MITRotor.ReferenceTurbines import IEA15MW
from optimise import NoControlBEM, JointControlBEM, YawControlBEM, ThrustControlBEM
from profilehooks import profile

from utilities import to_polars, from_polars

# Use Latex Fonts
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

REGENERATE = False
PARALLEL = False

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)

windfarm = Windfarm(rotor_model=BEM(IEA15MW(), BEM_model=DualBEM))
layout = Layout([0, 7], [0.0, 0.0])
wdirs = np.arange(-20, 20, 0.25)


methods = {
    "NoControl": NoControlBEM,
    "YawControl": YawControlBEM,
    "ThrustControl": ThrustControlBEM,
    "JointControl": JointControlBEM,
}


@profile(filename="prof.prof")
def _generate(x):
    method, wdir = x
    sol = methods[method](layout.rotate(wdir), windfarm).optimise(Cp_constraint=None)

    return to_polars(sol).with_columns(pl.lit(method).alias("method"), pl.lit(wdir).alias("wdir"))


@cache_polars(Path(__file__).parent.parent / "data/plot_01_2_turbine_wind_sweep_BEM.csv")
def generate(regenerate=False):
    params = list(product(methods, wdirs))

    dfs = foreach(_generate, params, parallel=PARALLEL)
    df = pl.concat(dfs)
    return df


def plot(df):
    fig, axes = plt.subplots(5, 1, sharex=True)

    for method in methods:
        _df = (
            df.filter(pl.col("method") == method)
            .group_by("wdir")
            .agg(
                pl.col("Cp").mean(),
                pl.col("setpoint_0").where(pl.col("turbine") == 0).first(),
                pl.col("setpoint_1").where(pl.col("turbine") == 0).first(),
                pl.col("setpoint_2").where(pl.col("turbine") == 0).first(),
                pl.col("Ctprime").where(pl.col("turbine") == 0).first(),
            )
        ).sort("wdir")

        axes[0].plot(_df["wdir"], _df["Cp"], label=method)
        axes[1].plot(_df["wdir"], np.rad2deg(_df["setpoint_0"]), label=method)
        axes[2].plot(_df["wdir"], _df["setpoint_1"], label=method)
        axes[3].plot(_df["wdir"], np.rad2deg(_df["setpoint_2"]), label=method)
        axes[4].plot(_df["wdir"], _df["Ctprime"], label=method)

    axes[-1].set_xlabel("wind direction [deg]")

    axes[0].set_ylabel("$C_P$")
    axes[1].set_ylabel(r"$\theta_p$ [deg]")
    axes[2].set_ylabel(r"$\lambda$")
    axes[3].set_ylabel(r"$\gamma$ [deg]")
    axes[4].set_ylabel(r"$C_T'$ [deg]")

    axes[0].set_ylim(0.4, 0.51)
    axes[0].legend()

    plt.savefig(FIGDIR / "wind_direction_sweep_BEM.png", dpi=300, bbox_inches="tight")


def plot_windfarms(df: pl.DataFrame):
    for (method, wdir), _df in df.group_by(["method", "wdir"]):
        windfarm_sol = from_polars(_df, windfarm)
        Plotting.plot_windfarm(windfarm_sol)
        plt.savefig(FIGDIR / f"{method}_{wdir}.png", dpi=300, bbox_inches="tight")
        plt.close()


def main():
    df = generate(regenerate=REGENERATE)
    plot(df)
    # plot_windfarms(df)


if __name__ == "__main__":
    main()
