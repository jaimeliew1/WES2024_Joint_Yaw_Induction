"""
Figure 9:   
Diamond setpoint distributions

This figure shows some kind of statistical patterns in the optimal setpoints. (what??)

Key points:
- ??
- Some patterns in the optimal setpoints
- Perhaps lower yaw spread in joint control strategy (but this isn't even the case?)
- ??
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from WES2024 import utils
from WES2024.Generate import diamond_AD

FILESTEM = Path(__file__).stem

group_palette = {
    "A": "tab:blue",
    "B": "tab:orange",
    "C": "tab:green",
    "D": "tab:red",
    "E": "tab:purple",
    "F": "tab:brown",
    "G": "tab:pink",
}


def generate(regenerate=False) -> pl.DataFrame:
    df = diamond_AD.generate(regenerate=regenerate)
    return df


def process(df_quarter: pl.DataFrame) -> pl.DataFrame:
    df = utils.fill_in_other_quadrants(df_quarter)
    return df


def plot(df: pl.DataFrame):
    plt.figure()
    _df = df.with_columns(np.rad2deg(pl.col("yaw")))

    graph = sns.jointplot(
        _df.to_pandas(),
        x="yaw",
        y="Ctprime",
        hue="group",
        # kind="scatter",
        ratio=3,
        height=4,
        palette=group_palette,
        # palette="viridis",
        s=7,
        edgecolors=None,
        legend=True,
        # marginal_kws=dict(bins=100),
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
    df = (
        generate(regenerate=False)
        .filter(pl.col("min_dist") == 6.0)
        .filter(pl.col("method") == "JointControl")
        .filter(pl.col("wdir") < 90.0)
    )
    df_proc = process(df)
    plot(df_proc)


if __name__ == "__main__":
    main()
