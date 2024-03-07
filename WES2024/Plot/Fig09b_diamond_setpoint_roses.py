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
from WES2024.Generate import diamond_AD, diamond_BEM

FILESTEM = Path(__file__).stem


def generate(regenerate=False) -> pl.DataFrame:
    df_AD = (
        diamond_AD.generate(regenerate=regenerate)
        .filter(pl.col("min_dist") == 6.0)
        .filter(pl.col("method") == "JointControl")
        .with_columns(type=pl.lit("AD"))
        .select(pl.exclude("setpoint_0", "setpoint_1", "min_dist"))
    )

    df_BEM = (
        diamond_BEM.generate(regenerate=regenerate)
        .filter(pl.col("method") == "JointControl")
        .with_columns(type=pl.lit("BEM"))
        .select(df_AD.columns)
    )

    turbines_to_keep = utils.DIAMOND_GROUPS.filter(pl.col("face") == 3)["turbine"].to_numpy()

    df_layout = (
        df_AD.filter(pl.col("wdir") == 0)
        .select("turbine", "x", "y")
        .join(utils.DIAMOND_GROUPS.select("turbine", "group"), on="turbine")
        .unique()
        .sort("turbine")
    )
    df = pl.concat(
        [utils.fill_in_other_quadrants(df_AD), utils.fill_in_other_quadrants(df_BEM)]
    ).filter(pl.col("turbine").is_in(turbines_to_keep))

    return df, df_layout


def plot(df: pl.DataFrame, df_layout: pl.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=np.array([6, 4]))

    # Plot wind farm layout
    for ax in axes.ravel():
        ax.plot(
            df_layout["x"],
            df_layout["y"],
            "o",
            ms=3,
            markerfacecolor="None",
            markeredgecolor="k",
            markeredgewidth=1,
            zorder=300,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("equal")

        ax.set_xlim(-8, 15)
        ax.set_ylim(5, 12)

    for ax, key in zip(axes.ravel(), ["yaw", "Ctprime"]):
        for group in df["group"].unique().sort():
            _df = df.filter(pl.col("group") == group).filter(pl.col("type") == "AD").sort("wdir")
            turbine = _df["turbine"].unique()[0]
            x, y = df_layout.filter(pl.col("turbine") == turbine).select("x", "y").row(0)
            utils.my_polar_plot(np.deg2rad(_df["wdir"]), _df[key], x=x, y=y, r0=0.2, width=5, style="-g", ax=ax)
    # Add legend
    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df, df_layout = generate(regenerate=False)

    plot(df, df_layout)


if __name__ == "__main__":
    main()
