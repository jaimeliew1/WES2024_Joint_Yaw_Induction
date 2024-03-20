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

from WES2024 import utils
from WES2024.Generate import diamond_AD_sensitivity

FILESTEM = Path(__file__).stem


def generate(regenerate=False) -> pl.DataFrame:
    df = diamond_AD_sensitivity.generate(regenerate=regenerate)
    return df.filter(pl.col("min_dist") == 6.0)


def plot(df_full: pl.DataFrame):

    fig, axes = plt.subplots(8, 1, sharex=True, sharey=True)
    turbines_to_keep = utils.DIAMOND_GROUPS.filter(pl.col("face") == 0)["turbine"]
    df = df_full.filter(pl.col("turbine").is_in(turbines_to_keep))

    for (group, _df), ax in zip(df.sort("group").group_by("group", maintain_order=True), axes):
        for method, __df in _df.sort("method").group_by("method", maintain_order=True):
            __df = __df.sort("wdir")
            ax.plot(
                __df["wdir"],
                __df["dCpdwdir"],
                label=utils.controller_labels[method],
                c=utils.controller_colors[method],
            )
            print(group, method, __df["dCpdwdir"].max())

    for method, _df in df_full.sort("method").group_by("method", maintain_order=True):

        _df = _df.group_by("wdir").agg(pl.col("dCpdwdir").mean()).sort("wdir")
        axes[-1].plot(
            _df["wdir"],
            _df["dCpdwdir"],
            label=utils.controller_labels[method],
            c=utils.controller_colors[method],
        )

    plt.xlim(0, 360)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4)
    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df_quarter = generate(regenerate=False)
    df = utils.fill_in_other_quadrants(df_quarter)

    # select only the first turbine in each group

    plot(df)


if __name__ == "__main__":
    main()
