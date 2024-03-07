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
import polars as pl
import seaborn as sns

from WES2024 import utils
from WES2024.Generate import diamond_AD

FILESTEM = Path(__file__).stem


def generate(regenerate=False) -> pl.DataFrame:
    df = diamond_AD.generate(regenerate=regenerate)
    return df.filter(pl.col("min_dist") == 6.0)


def plot(df: pl.DataFrame):
    df_agg = df.group_by("group", "method").agg(pl.col("Cp").mean())
    df_ref = df_agg.filter(pl.col("method") == "NoControl")

    df_agg = (
        df_agg.join(df_ref, on="group")
        .select("group", "method", (100 * (pl.col("Cp") / pl.col("Cp_right") - 1)).alias("Cp"))
        .filter(pl.col("method") != "NoControl")
        .sort("group")
    )

    sns.barplot(
        df_agg,
        x="group",
        y="Cp",
        hue="method",
        hue_order=["ThrustControl", "YawControl", "JointControl"],
    )

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df_quarter = generate(regenerate=False)
    df = utils.fill_in_other_quadrants(df_quarter)

    # select only the first turbine in each group
    turbines_to_keep = utils.DIAMOND_GROUPS.filter(pl.col("face") == 0)["turbine"]
    df = df.filter(pl.col("turbine").is_in(turbines_to_keep)).with_columns(pl.col("wdir").round(2))

    plot(df)


if __name__ == "__main__":
    main()
