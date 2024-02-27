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
import polars as pl
import seaborn as sns
import numpy as np

from WES2024 import utils, Fig07_diamond_wdir_sweep_many_spacings


FILESTEM = Path(__file__).stem


def generate(regenerate=False) -> pl.DataFrame:
    df = Fig07_diamond_wdir_sweep_many_spacings.generate(regenerate)
    return df


def plot(df: pl.DataFrame):
    plt.figure()
    _df = df.with_columns(np.rad2deg(pl.col("yaw"))).filter(
        pl.col("method").is_in(["YawControl", "JointControl"])
    )

    # sns.violinplot(
    #     _df.to_pandas(),
    #     # x="min_dist",
    #     y="yaw",
    #     hue="method",
    #     split=True,
    #     # inner=None,
    #     ax=plt.gca(),
    # )
    print(_df.columns)
    _df = _df.filter(pl.col("method") == "JointControl")  # .filter(pl.col("min_dist").is_in([4]))
    sns.jointplot(
        _df.to_pandas(),
        x="yaw",
        y="Ctprime",
        hue="min_dist",
        kind="scatter",
        ratio=3,
        height=4,
        palette="viridis",
        s=7,
    )

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=False)
    plot(df)


if __name__ == "__main__":
    main()
