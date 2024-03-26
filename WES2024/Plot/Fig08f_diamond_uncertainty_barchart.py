"""
Figure 4:
2 turbine LES

I am not sure what this figure will do, but it should include results from the
LES case.

Key points:
- ??

"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from WES2024 import utils
from WES2024.Generate import diamond_BEM_uncertainty_postproc, diamond_BEM

FILESTEM = Path(__file__).stem

turbines_to_keep = [17, 20, 21, 22, 16, 12]


def generate(regenerate=False) -> pl.DataFrame:
    df_quarter = diamond_BEM_uncertainty_postproc.generate(regenerate=regenerate)
    df = utils.fill_in_other_quadrants(df_quarter)

    df_bem_quarter = diamond_BEM.generate(regenerate=regenerate)
    df_bem = utils.fill_in_other_quadrants(df_bem_quarter)

    return df, df_bem


def plot(df: pl.DataFrame, df_bem: pl.DataFrame):
    fig, axes = plt.subplots(1, 2, width_ratios=[1, 1 / 8], figsize=0.8 * np.array([10, 4]))
    plt.subplots_adjust(wspace=0.3)
    df_turbines = (
        df.filter(pl.col("turbine").is_in(turbines_to_keep))
        .group_by("std", "group", "method")
        .agg(pl.col("Cp").mean())
    )

    # I think the hues are wrong
    g = sns.FacetGrid(
        df_turbines.sort("group", "method"), col="group", col_wrap=3, height=2, aspect=1
    )
    g.map_dataframe(sns.lineplot, x="std", hue="method", y="Cp")
    g.add_legend()

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")
    plt.close()
    df_agg_farm = df.group_by("method", "std").agg(pl.col("Cp").mean())

    print(df_agg_farm)

    sns.lineplot(
        df_agg_farm.sort("method"),
        x="std",
        y="Cp",
        hue="method",
        # hue_order=["NoControl", "ThrustControl", "YawControl", "JointControl"],
        palette=utils.controller_colors,
        ax=axes[0],
        legend=True,
    )
    plt.savefig(utils.FIGDIR / f"{FILESTEM}_farm.png", dpi=300, bbox_inches="tight")
    plt.close()

    df_agg_farm = (
        df.filter(
            pl.col("wdir") > 48 - 10,
            pl.col("wdir") < 48 + 10,
            pl.col("method").is_in(["NoControl", "YawControl", "JointControl"]),
        )
        .group_by("method", "std", "wdir")
        .agg(pl.col("Cp").mean())
    )

    print(df_agg_farm)

    plt.figure()
    sns.lineplot(
        df_agg_farm.sort("method"),
        x="wdir",
        y="Cp",
        size="std",
        hue="method",
        # hue_order=["NoControl", "ThrustControl", "YawControl", "JointControl"],
        palette=utils.controller_colors,
        # ax=axes[0],
        legend=True,
        sizes=(0.25, 2.5),
    )
    plt.savefig(utils.FIGDIR / f"{FILESTEM}_farm_by_wdir.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df, df_bem = generate(regenerate=False)
    plot(df, df_bem)


if __name__ == "__main__":
    main()
