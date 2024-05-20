"""
Figure 12b: mini wind roses.

Key points: - ??


"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from MITRotor.ReferenceTurbines import IEA15MW
from mitwindfarm import BEM, Plotting, Windfarm

from WES2024 import utils
from WES2024.BEM_gradients import DualBEM
from WES2024.Generate import diamond_BEM

FILESTEM = Path(__file__).stem

REGENERATE = False


windfarm = Windfarm(rotor_model=BEM(IEA15MW(), BEM_model=DualBEM))


def generate(regenerate=False):
    return diamond_BEM.generate(regenerate=regenerate)


def plot(df: pl.DataFrame):
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

    windfarm_sol = utils.from_polars(_df_to_plot, windfarm)
    Plotting.plot_windfarm(windfarm_sol, ax=ax, pad=10.0, res=1500)

    for i, (x, y, _) in enumerate(windfarm_sol.layout):
        _df = (
            df.filter(pl.col("turbine") == i)
            .filter(pl.col("method") == "JointControl")
            .select(np.deg2rad(pl.col("wdir")), pl.col("pitch"), pl.col("tsr"), pl.col("yaw"))
            .sort("wdir")
        )

        utils.my_polar_plot(_df["wdir"], _df["pitch"], x=x, y=y, r0=0.2, width=5, ax=ax, style="r-")
        utils.my_polar_plot(_df["wdir"], _df["tsr"], x=x, y=y, r0=0.2, width=5, ax=ax, style="g-")
        utils.my_polar_plot(_df["wdir"], _df["yaw"], x=x, y=y, r0=0.2, width=5, ax=ax, style="b-")

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=1000, bbox_inches="tight")


def main():
    df = generate(regenerate=REGENERATE)
    plot(df)


if __name__ == "__main__":
    main()
