"""
Figure 3:
2 turbine wind direction sweep

This figure shows the optimal setpoints (Cp, yaw, pitch, TSR, Ct' ) using four
different control strategies (None, yaw, thrust, and joint control) and two
different modelling methods (AD and BEM).

Key points:
- Joint control performs better than all other methods.
- AD and BEM show good agreement in setpoints. Losses exist in BEM.
- Yaw control is discontinuous.
- Joint control is smooth.
- Yaw control has non-constant thrust. i.e. yaw and thrust are coupled quantities.

"""
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from WES2024 import utils
from WES2024.Generate import two_turbine_AD_two_spacings

REGENERATE = False
FILESTEM = Path(__file__).stem

def generate(regenerate=False) -> pl.DataFrame:
    df = two_turbine_AD_two_spacings.generate(regenerate=regenerate).with_columns(
        pl.col("yaw").degrees()
    )
    return df

def plot(df):
    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(4, 3), sharex=True, sharey=True, gridspec_kw={'hspace': 0.5})
    ax[0].set_title("$s/D = 6$")
    ax[1].set_title("$s/D = 10$")
    # ax[0].set_xlabel("Wind direction (°)")
    ax[1].set_xlabel("Wind direction (°)")
    # ax[0].set_ylabel("Yaw misalignment angle (°)")

    df = df.group_by("wdir", "spacing", "method").agg(
                pl.col("yaw").where(pl.col("turbine") == 0).first(),
            ).sort("wdir")


    df1 = df.filter(pl.col("spacing") == 6, pl.col("method") == "YawControl").sort("wdir").with_columns(
        pl.when(
            pl.col("method").is_in(["YawControl", "YawKOmegaControl"]), pl.col("wdir").abs() < 1e-1
        )
        .then(np.nan)
        .otherwise(pl.col("yaw"))
        .alias("yaw"),)
    df2 = df.filter(pl.col("spacing") == 10, pl.col("method") == "YawControl").sort("wdir").with_columns(
        pl.when(
            pl.col("method").is_in(["YawControl", "YawKOmegaControl"]), pl.col("wdir").abs() < 1e-1
        )
        .then(np.nan)
        .otherwise(pl.col("yaw"))
        .alias("yaw"),)
    df3 = df.filter(pl.col("spacing") == 6, pl.col("method") == "JointControl").sort("wdir")
    df4 = df.filter(pl.col("spacing") == 10, pl.col("method") == "JointControl").sort("wdir").with_columns(
        pl.when(
            pl.col("method").is_in(["JointControl", "YawKOmegaControl"]), pl.col("wdir").abs() < 1e-1
        )
        .then(np.nan)
        .otherwise(pl.col("yaw"))
        .alias("yaw"),)

    ax[0].plot(df1["wdir"], df1["yaw"], label ="Yaw Control", c=plt.cm.tab20(2 / 20))
    ax[1].plot(df2["wdir"], df2["yaw"], label ="Yaw Control", c=plt.cm.tab20(2 / 20))
    ax[0].plot(df3["wdir"], df3["yaw"], label ="Joint Control", c=plt.cm.tab20(4 / 20))
    ax[1].plot(df4["wdir"], df4["yaw"], label ="Joint Control", c=plt.cm.tab20(4 / 20))
    ax[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.text(0.0, 0.5, 'Yaw misalignment angle, $\gamma$ ($^\circ$)', va='center', rotation='vertical')


    plt.savefig(utils.FIGDIRFORPAPER / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=REGENERATE)
    plot(df)


if __name__ == "__main__":
    main()
