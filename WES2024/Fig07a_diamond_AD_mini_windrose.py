"""
Figure 12b: mini wind roses.

Key points: - ??

DOES THE BEM CODE GIVE THE SAME SET POINTS IF U = 0.5 or U = 1??????????????
CHECK ON MONDAY!!! 
I NEED TO NORMALISE BASED ON U0 (REWS at the turbine location, but without the
turbine), NOT U_\inf!!
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mitwindfarm import Plotting
from MITRotor.ReferenceTurbines import IEA15MW
from mitwindfarm.Rotor import BEM
from mitwindfarm.windfarm import Windfarm
from test_BEM_gradients import DualBEM

from WES2024 import Fig07_diamond_wdir_sweep_many_spacings, utils

FILESTEM = Path(__file__).stem

REGENERATE = False


windfarm = Windfarm(rotor_model=BEM(IEA15MW(), BEM_model=DualBEM))


def generate(regenerate=False):
    return Fig07_diamond_wdir_sweep_many_spacings.generate(regenerate=regenerate)


def plot(df_quarter: pl.DataFrame):
    plt.figure()
    ax = plt.gca()

    df_quarter = (
        df_quarter.filter(pl.col("method").is_in(["JointControl"]))
        .filter(pl.col("min_dist").is_in([6.0]))
        .with_columns(
            np.rad2deg(pl.col("yaw")),
        )
        # .select(pl.exclude("turbine"))
    )
    df = utils.fill_in_other_quadrants(df_quarter)

    _df_to_plot = df_quarter.filter(pl.col("wdir") == 0.0)
    print(_df_to_plot)
    windfarm_sol = utils.from_polars(_df_to_plot, windfarm)
    Plotting.plot_windfarm(windfarm_sol, ax=ax, pad=10.0, res=1000)

    for i, (x, y, _) in enumerate(windfarm_sol.layout):
        _df = (
            df.filter(pl.col("turbine") == i)
            .filter(pl.col("method") == "JointControl")
            .select(np.deg2rad(pl.col("wdir")), pl.col("Ctprime"), pl.col("yaw"))
            .sort("wdir")
        )

        # plot turbine number
        ax.text(x, y, f"{i+1}", c="r")

        R = 0.75

        rose_x = (_df["Ctprime"] + 0) * R * (-np.cos(_df["wdir"])) + x
        rose_y = (_df["Ctprime"] + 0) * R * (np.sin(_df["wdir"])) + y
        ax.plot(rose_x, rose_y, "g-", lw=0.5)

        R = 0.02
        rose_x = (_df["yaw"] + 90) * R * (-np.cos(_df["wdir"])) + x
        rose_y = (_df["yaw"] + 90) * R * (np.sin(_df["wdir"])) + y
        ax.plot(rose_x, rose_y, "b-", lw=0.5)

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=1000, bbox_inches="tight")


def main():
    df_quarter = generate(regenerate=REGENERATE)

    plot(df_quarter)


if __name__ == "__main__":
    main()
