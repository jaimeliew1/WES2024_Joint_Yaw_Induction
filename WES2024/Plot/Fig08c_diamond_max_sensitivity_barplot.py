"""
Figure 8:   
Diamond layout - power increase due to control versus turbine spacing

This figure shows how much power is gained by performing optimal wind farm
control (versus not performing control) versus turbine spacing. Assumes uniform
wind rose.

Key points:
- Yaw control outperforms thrust control (known in literature).
- Joint control outperforms yaw control (novel result).
- Benefit of control is larger for for close spacings.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from rich import print

from WES2024 import utils
from WES2024.Generate import diamond_AD_sensitivity

FILESTEM = Path(__file__).stem


def generate(regenerate=False) -> pl.DataFrame:
    df = diamond_AD_sensitivity.generate(regenerate=regenerate)
    return df.filter(pl.col("min_dist") == 6.0)


def plot(df: pl.DataFrame):

    fig, axes = plt.subplots(1, 2, width_ratios=[1, 1 / 8], figsize=0.8 * np.array([10, 4]))
    plt.subplots_adjust(wspace=0.3)
    df_agg = df.group_by("group", "method").agg(pl.col("dCpdwdir").abs().max()).sort("group")
    print(df_agg)

    sns.barplot(
        df_agg,
        x="group",
        y="dCpdwdir",
        hue="method",
        hue_order=["NoControl", "ThrustControl", "YawControl", "JointControl"],
        palette=utils.controller_colors,
        ax=axes[0],
    )
    df_agg_farm = (
        df.group_by("method", "wdir")
        .mean()
        .group_by("method")
        .agg(pl.col("dCpdwdir").abs().max())
        .with_columns(pl.lit("Farm").alias("group"))
    )
    print(df_agg_farm)

    sns.barplot(
        df_agg_farm,
        x="group",
        y="dCpdwdir",
        hue="method",
        hue_order=["NoControl", "ThrustControl", "YawControl", "JointControl"],
        palette=utils.controller_colors,
        ax=axes[1],
        legend=False,
    )
    axes[0].set_ylabel(
        r"$\max\left(\left|\frac{\partial C_P}{\partial \alpha}\right|\right) $(deg$^{-1}$)"
    )
    axes[1].set_ylabel(
        r"$\max\left(\left|\frac{\partial C_{P,\mathrm{farm}}}{\partial \alpha}\right|\right) $(deg$^{-1}$)"
    )

    sns.move_legend(
        axes[0],
        "lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        # frameon=False,
        # fontsize="xx-small",
        title=None,
    )
    # axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.01))
    # Set same y lim on both axes
    axes[1].set_ylim(*axes[0].get_ylim())
    axes[1].set_xlabel("")

    axes[0].set_xlabel("Turbine")

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df_quarter = generate(regenerate=False)
    df = utils.fill_in_other_quadrants(df_quarter)

    plot(df)


if __name__ == "__main__":
    main()
