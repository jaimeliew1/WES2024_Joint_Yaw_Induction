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
from WES2024.Generate import diamond_AD, diamond_BEM

FILESTEM = Path(__file__).stem


def generate(regenerate=False) -> pl.DataFrame:
    # df = diamond_AD.generate(regenerate=regenerate)
    # return df.filter(pl.col("min_dist") == 6.0)
    df = diamond_BEM.generate(regenerate=regenerate)
    return df


def aggregate_turbine(df_full: pl.DataFrame) -> tuple[pl.DataFrame, ...]:
    # select only the first turbine in each group
    turbines_to_keep = utils.DIAMOND_GROUPS.filter(pl.col("face") == 0, pl.col("group") != "Dinv")[
        "turbine"
    ]
    df = df_full.filter(pl.col("turbine").is_in(turbines_to_keep))

    df_agg = df.group_by("group", "method").agg(pl.col("Cp").mean(), pl.col("Ct").mean())
    df_ref = df_agg.filter(pl.col("method") == "NoControl")

    df_agg = (
        df_agg.join(df_ref, on="group")
        .select(
            "group",
            "method",
            (100 * (pl.col("Cp") / pl.col("Cp_right") - 1)).alias("Cp"),
            (100 * (pl.col("Ct") / pl.col("Ct_right") - 1)).alias("Ct"),
        )
        .filter(pl.col("method") != "NoControl")
        .sort("group")
    )
    df_farm_agg = df_full.group_by("method").agg(pl.col("Cp").mean(), pl.col("Ct").mean())
    Cp_farm_ref = df_farm_agg.filter(pl.col("method") == "NoControl")["Cp"][0]
    Ct_farm_ref = df_farm_agg.filter(pl.col("method") == "NoControl")["Ct"][0]
    df_farm_agg = df_farm_agg.with_columns(
        (100 * (pl.col("Cp") / Cp_farm_ref - 1)).alias("Cp"),
        (100 * (pl.col("Ct") / Ct_farm_ref - 1)).alias("Ct"),
        pl.lit("Farm").alias("group"),
    )

    return df_agg, df_farm_agg


def plot(df_turb: pl.DataFrame, df_farm: pl.DataFrame):
    fig, axes = plt.subplots(2, 2, width_ratios=[1, 1 / 8], figsize=0.8 * np.array([10, 4]))
    plt.subplots_adjust(wspace=0.25)

    sns.barplot(
        df_turb,
        x="group",
        y="Cp",
        hue="method",
        hue_order=["ThrustControl", "YawControl", "JointControl"],
        palette=utils.controller_colors,
        ax=axes[0, 0],
        legend=True,
    )

    sns.barplot(
        df_farm,
        x="group",
        y="Cp",
        hue="method",
        hue_order=["ThrustControl", "YawControl", "JointControl"],
        palette=utils.controller_colors,
        ax=axes[0, 1],
        legend=False,
    )

    sns.barplot(
        df_turb,
        x="group",
        y="Ct",
        hue="method",
        hue_order=["ThrustControl", "YawControl", "JointControl"],
        palette=utils.controller_colors,
        ax=axes[1, 0],
        legend=False,
    )

    sns.barplot(
        df_farm,
        x="group",
        y="Ct",
        hue="method",
        hue_order=["ThrustControl", "YawControl", "JointControl"],
        palette=utils.controller_colors,
        ax=axes[1, 1],
        legend=False,
    )

    axes[0, 0].set_ylabel(r"$C_P$ increase (%)")
    axes[1, 0].set_ylabel(r"$C_T$ increase (%)")
    axes[0, 1].set_ylabel(r"   $C_{P,\mathrm{farm}}$ increase (%)")
    axes[1, 1].set_ylabel(r"$C_{T,\mathrm{farm}}$ increase (%)   ")

    sns.move_legend(
        axes[0, 0],
        "lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        title=None,
    )

    # Set same y lim on both axes
    axes[0, 0].set_ylim(-1, 8)
    axes[0, 1].set_ylim(-1, 8)
    axes[1, 0].set_ylim(-6, 7)
    axes[1, 1].set_ylim(-6, 7)

    # Set axis labels and ticks
    axes[0, 0].set_xlabel("")
    axes[0, 1].set_xlabel("")
    axes[1, 1].set_xlabel("")
    axes[0, 0].set_xticklabels([])
    axes[0, 1].set_xticklabels([])
    axes[1, 1].set_xticklabels([])
    axes[1, 0].set_xlabel("Turbine")
    axes[1, 1].set_xlabel("Farm")

    plt.savefig(utils.FIGDIRFORPAPER / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df_full = utils.fill_in_other_quadrants(generate(regenerate=False))

    df_turb, df_farm = aggregate_turbine(df_full)

    print(df_turb)
    print(df_farm)

    plot(df_turb, df_farm)


if __name__ == "__main__":
    main()
