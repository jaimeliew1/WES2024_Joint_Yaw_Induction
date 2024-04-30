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
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mitwindfarm.Layout import Square

from WES2024 import utils
from WES2024.Generate import diamond_BEM

FILESTEM = Path(__file__).stem
# Diamond layout, translated to have a zero centroid
LAYOUT = Square(10.0, 5).rotate(45)
LAYOUT.x -= LAYOUT.x.mean()
LAYOUT.y -= LAYOUT.y.mean()

# minibox
box_xy = (-32, -5)
box_width = 37
box_height = 25

# turbines_to_keep = utils.DIAMOND_GROUPS.filter(pl.col("face") == 3)["turbine"].to_numpy()
turbines_to_keep = [17, 20, 21, 22, 16, 12]
print(turbines_to_keep)

titles = {
    "Ctprime": r"$C_T'$",
    "yaw": r"$\gamma$",
    "pitch": r"$\theta_p$",
    "tsr": r"$\lambda$",
}


def generate(regenerate=False) -> pl.DataFrame:
    df_quarter = diamond_BEM.generate(regenerate=regenerate).with_columns(
        pl.col("setpoint_0").alias("pitch"), pl.col("setpoint_1").alias("tsr")
    )
    df = utils.fill_in_other_quadrants(df_quarter)

    return df


def plot_layout_and_minirose(df: pl.DataFrame, channel: str, ax: plt.Axes):
    """
    Plot the wind farm layout and mini power/setpoint roses over selected turbines.

    Args:
        df (pl.DataFrame): A DataFrame containing wind direction, method, turbine, and channel data.
        channel (str): The column name in `df` representing the channel to plot (e.g., "Cp", "Ct", etc.).
        ax (plt.Axes): The axes object to plot on.

    This function plots the layout of the wind farm on the given `ax` axes.
    Additionally, it plots mini power/setpoint roses for a subset of turbines
    specified by `turbines_to_keep`, showing the values of the specified
    `channel` for each turbine at different wind directions.
    """
    ax.plot(
        LAYOUT.x,
        LAYOUT.y,
        "o",
        ms=3,
        markerfacecolor="None",
        markeredgecolor="k",
        markeredgewidth=1,
        zorder=300,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    for turb_no in turbines_to_keep:
        x, y = LAYOUT.x[turb_no], LAYOUT.y[turb_no]
        _df = df.filter(turbine=turb_no, method="JointControl").sort("wdir")

        # Plot mini wind rose with a given radius, width, and other line parameters.
        utils.my_polar_plot(
            np.deg2rad(_df["wdir"]),
            _df[channel],
            x=x,
            y=y,
            r0=0.3,
            width=5,
            ax=ax,
            c="tab:red",
            lw=0.5,
        )


def plot_layout_and_powerrose(
    df: pl.DataFrame, fig: plt.Figure, axp: plt.Axes
) -> tuple[plt.Axes, plt.Axes]:
    """
    Create a plot with a power/setpoint rose over wind farm layout.

    Args:
        df (pl.DataFrame): A DataFrame containing wind direction (wdir), method, and power coefficient (Cp) data.
        fig (plt.Figure): The figure object to create the plot in.
        axp (plt.Axes): The axes object to create the power rose in.

    Returns:
        tuple[plt.Axes, plt.Axes]: A tuple containing the polar and cartesian axes.

    This function creates a plot with a power/wind rose for different control
    methods at different wind directions, and a layout plot showing the
    positions of the wind turbines in the farm. The power/setpoint rose is
    plotted on a polar axis (axp), while layout is plotted on an overlapping
    cartesian axes (ax).
    """
    _df = df.pivot(index="wdir", columns="method", values="Cp", aggregate_function="mean").sort(
        "wdir"
    )

    # Overlay a cartesian axes on the existing polar axis.
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

    # Plot the large power rose over the wind farm
    wdirs = np.deg2rad(_df["wdir"])
    for method in ["NoControl", "ThrustControl", "YawControl", "JointControl"]:
        axp.plot(np.pi / 2 + wdirs, _df[method], lw=1, **utils.line_params[method])

    xs, ys = LAYOUT.x, LAYOUT.y

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
    fig, axd = plt.subplot_mosaic(
        [["polar", "A", "B"], ["polar", "C", "D"]],
        per_subplot_kw={"polar": {"projection": "polar"}},
        width_ratios=[0.3, 0.25, 0.25],
        figsize=0.8 * np.array((10, 5)),
    )
    plt.subplots_adjust(hspace=0.00)
    _, ax_layout = plot_layout_and_powerrose(df, fig, axd["polar"])
    axd["polar"].legend(ncol=2, loc="lower center", fontsize="xx-small", bbox_to_anchor=(0.5, 1.15))

    channels = ["Ctprime", "yaw", "pitch", "tsr"]
    for ax_id, channel in zip(["A", "B", "C", "D"], channels):
        plot_layout_and_minirose(df, channel, axd[ax_id])

        axd[ax_id].set_xlim(box_xy[0], box_xy[0] + box_width)
        axd[ax_id].set_ylim(box_xy[1], box_xy[1] + box_height)

        axd[ax_id].set_title(titles[channel])

        for spine in axd[ax_id].spines.values():
            spine.set_linestyle("--")
            spine.set_color("teal")

    rectangle = Rectangle(box_xy, box_width, box_height, fill=False, ls="--", ec="teal")
    ax_layout.add_patch(rectangle)

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=False)

    plot(df)


if __name__ == "__main__":
    main()
