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


groups = {
    "OC": (0, 4, 20, 24),
    "OLE1": (1, 15, 23, 9),
    "OLE2": (5, 21, 19, 3),
    "OCE": (2, 10, 22, 14),
    "IC": (8, 6, 16, 18),
    "IE": (7, 11, 17, 13),
    "C": (12,),
}

# Note: the center turbine (12) is not actually in face 0.
faces = {
    0: (0, 1, 2, 3, 6, 7, 12),
    1: (5, 10, 15, 20, 11, 16),
    2: (21, 22, 23, 24, 17, 18),
    3: (8, 13, 4, 9, 14, 19),
}

groups_map = {v: key for key, vals in groups.items() for v in vals}
face_map = {v: key for key, vals in faces.items() for v in vals}

group_palette = {
    "OC": "tab:blue",
    "OLE1": "tab:orange",
    "OLE2": "tab:green",
    "OCE": "tab:red",
    "IC": "tab:purple",
    "IE": "tab:brown",
    "C": "tab:pink",
}


def generate(regenerate=False):
    return Fig11_diamond_pitch_tsr_surface.generate(regenerate=regenerate)


def plot(df: pl.DataFrame):
    print(df.columns)
    plt.figure()
    df = (
        df.rename(dict(setpoint_0="pitch", setpoint_1="tsr")).filter(
            pl.col("method").is_in(["JointControl"])
        )
        # .filter(pl.col("method").is_in(["ThrustControl"]))
        # .filter(pl.col("method").is_in(["ThrustControl", "JointControl"]))
        .with_columns(
            np.rad2deg(pl.col("yaw")),
            np.rad2deg(pl.col("pitch")),
            pl.col("turbine").replace(groups_map).alias("group"),
        )  # .filter(pl.col("group") == "OLE2")
    )

    sns.jointplot(
        df.to_pandas(),
        # x="pitch",
        # y="tsr",
        y="Ctprime",
        x="yaw",
        hue="group",
        # kind="kde",
        ratio=3,
        height=4,
        palette=group_palette,
        legend=False,
        s=7,
        edgecolors=None,
    )

    # plt.ylim(0, 4)
    # plt.xlim(-50, 50)
    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=REGENERATE)
    plot(df)


if __name__ == "__main__":
    main()
