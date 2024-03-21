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


# Diamond layout, translated to have a zero centroid
layout = Square(10.0, 5).rotate(45)
layout.x -= layout.x.mean()
layout.y -= layout.y.mean()


def generate(regenerate=False) -> pl.DataFrame:
    df_quarter = diamond_AD.generate(regenerate=regenerate)
    df = utils.fill_in_other_quadrants(df_quarter)
    return df


def plot_layout_and_powerrose(
    df: pl.DataFrame, fig: plt.Figure, axp: plt.Axes
) -> tuple[plt.Axes, ...]:
    _df = (
        df.filter(pl.col("min_dist") == 7)
        .pivot(index="wdir", columns="method", values="Cp", aggregate_function="mean")
        .sort("wdir")
    )
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
        axp.plot(np.pi / 2 + wdirs, _df[method], **utils.line_params[method])

    xs, ys = layout.x, layout.y

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

    # add radial grid lines
    theta = np.linspace(0, np.pi * 2, 200)
    axp.plot(theta, np.zeros_like(theta), lw=0.7, ls="--", c="0.7")
    axp.plot(theta, 16 / 27 * np.ones_like(theta), lw=0.7, ls="--", c="0.7")
    axp.text(np.pi / 2 + 0.05, 0, r"$0$", c="0.7", fontsize=5, va="bottom")
    axp.text(np.pi / 2 + 0.05, 0.6, r"$0.6$", c="0.7", fontsize=5, va="bottom")

    ax.set_xlim(xs.min() * 2.2, xs.max() * 2.2)
    ax.set_ylim(ys.min() * 2.2, ys.max() * 2.2)

    return axp, ax


def plot(df: pl.DataFrame):
    fig = plt.figure(figsize=1.5 * np.array([2, 2]))
    axp = fig.add_subplot(frame_on=True, polar=True)

    axp, ax = plot_layout_and_powerrose(df, fig, axp)
    axp.legend(ncol=2, loc="lower center", fontsize="xx-small", bbox_to_anchor=(0.5, 1.10))

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df = generate(regenerate=False)
    plot(df)


if __name__ == "__main__":
    main()
