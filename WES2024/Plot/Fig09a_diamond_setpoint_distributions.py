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

    turbines_to_keep = utils.DIAMOND_GROUPS.filter(pl.col("face") == 0)["turbine"].to_numpy()

    df = pl.concat(
        [utils.fill_in_other_quadrants(df_AD), utils.fill_in_other_quadrants(df_BEM)]
    ).filter(pl.col("turbine").is_in(turbines_to_keep))

    return df


def plot(df: pl.DataFrame):
    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=np.array([4, 6]))

    for group, ax in zip(df["group"].unique().sort(), axes.ravel()):
        _df = df.filter(pl.col("group") == group).sort("wdir")
        sns.lineplot(
            _df,
            x="yaw",
            y="Ctprime",
            hue="type",
            ax=ax,
            sort=False,
            legend=group == "A",
        )
        # label turbine group
        ax.text(0.5, 0.98, f"Turbine {group}", ha="center", va="top", transform=ax.transAxes)

    # Remove individual axis labels
    for ax in axes.ravel():
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    # add axis lines at Ct'=2, yaw=0
    for ax in axes.ravel():
        ax.axhline(2, lw=0.75, c="0.75", zorder=0)
        ax.axvline(0, lw=0.75, c="0.75", zorder=0)

    # Add shared axis labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor="none", which="both", top=False, bottom=False, left=False, right=False
    )
    plt.xlabel(r"$\gamma$ (deg)")
    plt.ylabel(r"$C_T'$ (-)")

    # Add legend
    axes[0, 0].legend(loc="lower center", ncol=2, bbox_to_anchor=(1.05, 1.01))
    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=False)
    plot(df)


if __name__ == "__main__":
    main()
