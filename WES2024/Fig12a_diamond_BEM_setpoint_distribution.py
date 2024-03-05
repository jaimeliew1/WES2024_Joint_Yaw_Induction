"""
Figure 12:
Diamond BEM setpoint distribution

This figure shows how the optimal BEM setpoints are distributed for the Diamond
layout.

Key points:
- pitch and tsr follow minimum thrust trajectory (??)
- Some other nice geometric take aways??
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
import numpy as np
import polars as pl
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
    fig = plt.figure()

    df = df.rename(dict(setpoint_0="pitch", setpoint_1="tsr")).with_columns(
        np.rad2deg(pl.col("yaw")),
        np.rad2deg(pl.col("pitch")),
        np.rad2deg(np.abs(pl.col("yaw"))).alias("abs_yaw"),
    )
    norm = mpl.colors.Normalize(0, 15)

    graph = sns.scatterplot(
        df.to_pandas(),
        x="pitch",
        y="tsr",
        hue="abs_yaw",
        hue_norm=norm,
        palette="magma",
        legend=False,
        s=7,
        edgecolors=None,
        ax=plt.gca(),
    )
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="magma"),
        ax=plt.gca(),
        orientation="vertical",
        label="$|\gamma|$ [deg]",
    )
    graph.set_xlabel(r"$\theta_p$ [deg]")
    graph.set_ylabel(r"$\lambda$ [-]")
    
    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df_quarter = generate(regenerate=REGENERATE).filter(pl.col("method") == "JointControl")
    plot(df_quarter)


if __name__ == "__main__":
    main()
