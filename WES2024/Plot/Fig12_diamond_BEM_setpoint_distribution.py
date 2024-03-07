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

from WES2024 import utils
from WES2024.Generate import diamond_BEM

FILESTEM = Path(__file__).stem

REGENERATE = False


group_palette = {
    "A": "tab:blue",
    "B": "tab:orange",
    "C": "tab:green",
    "D": "tab:red",
    "E": "tab:purple",
    "F": "tab:brown",
    "G": "tab:pink",
}


def generate(regenerate=False):
    return diamond_BEM.generate(regenerate=regenerate)


def plot(df: pl.DataFrame):
    plt.figure()
    df = df.rename(dict(setpoint_0="pitch", setpoint_1="tsr")).with_columns(
        np.rad2deg(pl.col("yaw")), np.rad2deg(pl.col("pitch"))
    )

    graph = sns.jointplot(
        df.to_pandas(),
        y="Ctprime",
        x="yaw",
        hue="group",
        ratio=3,
        height=4,
        palette=group_palette,
        legend=True,
        s=7,
        edgecolors=None,
    )

    sns.move_legend(
        graph.ax_joint,
        "lower center",
        bbox_to_anchor=(0.5, 0.9),
        ncol=4,
        frameon=False,
        fontsize="xx-small",
        title=None,
    )

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df_quarter = generate(regenerate=REGENERATE).filter(pl.col("method") == "JointControl")
    df = utils.fill_in_other_quadrants(df_quarter)
    plot(df)


if __name__ == "__main__":
    main()
