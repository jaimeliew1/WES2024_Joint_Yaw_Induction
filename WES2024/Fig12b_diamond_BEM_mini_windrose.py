"""
Figure 12b: mini wind roses.

Key points: - ??

DOES THE BEM CODE GIVE THE SAME SET POINTS IF U = 0.5 or U = 1??????????????
CHECK ON MONDAY!!! I think this is okay. tsr is normalised by free wind speed,
and so is thrust. so things will change if wind speed changes.
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

from WES2024 import Fig11_diamond_pitch_tsr_surface, utils

FILESTEM = Path(__file__).stem

REGENERATE = False


windfarm = Windfarm(rotor_model=BEM(IEA15MW(), BEM_model=DualBEM))


def generate(regenerate=False):
    return Fig11_diamond_pitch_tsr_surface.generate(regenerate=regenerate)


def plot(df: pl.DataFrame):
    print(df.columns)
    plt.figure()
    ax = plt.gca()

    df_quarter = (
        df.rename(dict(setpoint_0="pitch", setpoint_1="tsr"))
        .filter(pl.col("method").is_in(["JointControl"]))
        .with_columns(
            np.rad2deg(pl.col("yaw")),
            np.rad2deg(pl.col("pitch")),
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
            .select(np.deg2rad(pl.col("wdir")), pl.col("pitch"), pl.col("tsr"), pl.col("yaw"))
            .sort("wdir")
        )

        R = 0.5 * 0.75

        rose_x = (_df["pitch"] + 10) * R * (-np.cos(_df["wdir"])) + x
        rose_y = (_df["pitch"] + 10) * R * (np.sin(_df["wdir"])) + y
        ax.plot(rose_x, rose_y, "r-", lw=0.5)

        rose_x = (_df["tsr"] + 0) * R * (-np.cos(_df["wdir"])) + x
        rose_y = (_df["tsr"] + 0) * R * (np.sin(_df["wdir"])) + y
        ax.plot(rose_x, rose_y, "g-", lw=0.5)

        R = 0.15 * 0.5
        rose_x = (_df["yaw"] + 50) * R * (-np.cos(_df["wdir"])) + x
        rose_y = (_df["yaw"] + 50) * R * (np.sin(_df["wdir"])) + y
        ax.plot(rose_x, rose_y, "b-", lw=0.5)

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=1000, bbox_inches="tight")


def main():
    df = generate(regenerate=REGENERATE)
    plot(df)


if __name__ == "__main__":
    main()
