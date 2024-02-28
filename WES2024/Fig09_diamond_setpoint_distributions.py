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


def generate(regenerate=False) -> pl.DataFrame:
    df = Fig07_diamond_wdir_sweep_many_spacings.generate(regenerate)
    return df


def process(df: pl.DataFrame) -> pl.DataFrame:
    _df = df.with_columns(
        pl.col("turbine").map_dict(groups_map).alias("group"),
        pl.col("turbine").map_dict(face_map).alias("face"),
    ).with_columns((pl.col("wdir") + 90 * pl.col("face")).alias("wdir"))

    return _df


def plot(df: pl.DataFrame):
    plt.figure()
    _df = df.with_columns(np.rad2deg(pl.col("yaw"))).filter(
        pl.col("method").is_in(["JointControl"])
    )

    print(_df.columns)
    _df = (
        _df
        # .filter(pl.col("method") == "JointControl")
        .filter(pl.col("min_dist").is_in([10]))
        # .filter(pl.col("group").is_in(["OC"]))
    )
    print(_df)
    sns.jointplot(
        _df.to_pandas(),
        x="yaw",
        y="Ctprime",
        hue="group",
        # kind="scatter",
        ratio=3,
        height=4,
        palette=group_palette,
        s=7,
        edgecolors=None,
        legend=False,
        # marginal_kws=dict(bins=100),
    )

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=False)
    df_proc = process(df)
    plot(df_proc)


if __name__ == "__main__":
    main()
