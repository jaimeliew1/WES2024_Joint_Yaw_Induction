"""
Figure 6:   
diamond wind farm layout with polar plot of farm Cp

Shows farm power output versus wind direction, with overlay of wind farm layout.
One line per control strategy. AD rotor only (so far)

Key points:
- Strong wake scenarios are when WFC is most beneficial.
- Joint control is best.
- Line and rotational and symmetry.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from mitwindfarm.Layout import Square

from WES2024 import utils
from WES2024.Generate import diamond_AD


FILESTEM = Path(__file__).stem

layout = Square(10.0, 5).rotate(45)


def generate(regenerate=False) -> pl.DataFrame:
    df = diamond_AD.generate(regenerate=regenerate)
    return df


styles = {
    "NoControl": dict(c="k", ls="--", lw=1, label="NoControl"),
    "ThrustControl": dict(c="tab:blue", lw=1, label="ThrustControl"),
    "YawControl": dict(c="tab:orange", lw=1, label="YawControl"),
    "JointControl": dict(c="tab:red", lw=1, label="JointControl"),
}


def plot(df: pl.DataFrame):
    _df = (
        df.filter(pl.col("min_dist") == 7)
        .pivot(index="wdir", columns="method", values="Cp", aggregate_function="mean")
        .sort("wdir")
    )

    fig = plt.figure(figsize=1.5 * np.array([2, 2]))
    axp = fig.add_subplot(frame_on=True, polar=True)
    ax = fig.add_axes(axp.get_position().bounds, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("equal")

    axp.set_theta_zero_location("W")
    axp.set_theta_direction(-1)
    axp.set_yticks([])
    axp.set_yticklabels([])
    axp.set_ylim(-0.7, 0.7)
    axp.grid(linestyle=":")

    wdirs = np.deg2rad(_df["wdir"])
    for method in ["NoControl", "ThrustControl", "YawControl", "JointControl"]:
        for sector in [0, 1, 2, 3]:
            axp.plot(sector * np.pi / 2 + wdirs, _df[method], **styles[method])
            styles[method]["label"] = None

    xs, ys = layout.x, layout.y
    xs -= xs.mean()
    ys -= ys.mean()
    ax.plot(
        xs,
        ys,
        "o",
        ms=3,
        markerfacecolor="None",
        markeredgecolor="k",
        markeredgewidth=1,
        zorder=300,
    )
    ax.set_xlim(xs.min() * 2, xs.max() * 2)
    ax.set_ylim(ys.min() * 2, ys.max() * 2)

    axp.legend(ncol=2, loc="lower center", fontsize="xx-small", bbox_to_anchor=(0.5, 1.10))

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df = generate(regenerate=False)
    plot(df)


if __name__ == "__main__":
    main()
