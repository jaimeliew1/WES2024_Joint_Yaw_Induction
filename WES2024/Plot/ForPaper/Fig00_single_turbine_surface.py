from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from WES2024.Generate import pitch_tsr_surface, minCt_trajectory
from WES2024 import utils


FILESTEM = Path(__file__).stem
# Use Latex Fonts
# plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


PITCHES = np.deg2rad(np.arange(-15, 15.001, 0.5))
TSRS = np.arange(5, 12.001, 0.125)
# YAW2 = 50  # deg
YAW2 = 45  # deg

XLIM = (-15, 15)
YLIM = (5, 10.5)


def generate(regenerate=False):
    df_surface = (
        pitch_tsr_surface.generate(regenerate=regenerate)
        .with_columns(
            pl.col("setpoint_0").alias("pitch"),
            pl.col("setpoint_1").alias("tsr"),
            pl.col("yaw"),
        )
        .with_columns(
            pl.col("pitch").degrees(),
            pl.col("yaw").degrees(),
        )
        .with_columns(type=pl.lit("surface"))
    )
    df_trajectory = (
        minCt_trajectory.generate(regenerate=regenerate)
        .with_columns(
            pl.col("pitch").degrees(),
        )
        .with_columns(type=pl.lit("trajectory"))
    )

    return df_surface, df_trajectory


def plot_surface(
    df: pl.DataFrame,
    x: str,
    y: str,
    z: str,
    ax: plt.Axes,
    levels=None,
    aggregate_function=None,
    **kwargs,
):
    ## Plot surface
    df_piv_Cp = df.pivot(index=y, columns=x, values=z, aggregate_function=aggregate_function)

    Y = df_piv_Cp[y].to_numpy()
    X = np.array(df_piv_Cp.columns[1:], dtype=float)
    Z = df_piv_Cp.to_numpy()[:, 1:]
    Z[Z < 0] = 0.01
    CF = ax.contourf(X, Y, Z, levels=levels, **kwargs)
    CS = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.8)
    ax.clabel(CS, inline=True, fontsize=10)

    return CF


def plot(df_surface: pl.DataFrame, df_trajectory: pl.DataFrame):
    # fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(8, 4))
    fig, axes = plt.subplots(2, 3, width_ratios=[1, 1, 0.1], figsize=np.array((8, 4)))

    # Share axes for contour plots only (exclude the color bar axes)
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.sharex(axes[0, 0])
        ax.sharey(axes[0, 0])

    # Remove tick labels on interior axes
    axes[0, 0].tick_params(labelbottom=False)
    axes[0, 1].tick_params(labelbottom=False)

    axes[0, 1].tick_params(labelleft=False)
    axes[1, 1].tick_params(labelleft=False)

    # Set axes limits
    axes[0, 0].set_xlim(*XLIM)
    axes[0, 0].set_ylim(*YLIM)

    # filter data to fit axis limits
    df_surface = df_surface.filter(pl.col("tsr") <= YLIM[1] * 1.01)

    # Axis labels
    [ax.set_xlabel(r"Pitch, $\theta_p~$(deg) ") for ax in axes[1, :2]]
    [ax.set_ylabel(r"Tip Speed Ratio, $\lambda~$(-)") for ax in axes[:, 0]]

    # Plot surfaces
    levels = np.arange(0, 0.60, 0.05)
    CF_Cp = plot_surface(
        df_surface.filter(yaw=0.0),
        "pitch",
        "tsr",
        "Cp",
        ax=axes[0, 0],
        levels=levels,
        cmap="viridis",
    )
    plot_surface(
        df_surface.filter(yaw=45.0),
        "pitch",
        "tsr",
        "Cp",
        ax=axes[0, 1],
        levels=levels,
        cmap="viridis",
    )

    levels = np.arange(0, 2, 0.1)
    CF_Ct = plot_surface(
        df_surface.filter(yaw=0.0),
        "pitch",
        "tsr",
        "Ct",
        ax=axes[1, 0],
        levels=levels,
        cmap="plasma",
    )
    plot_surface(
        df_surface.filter(yaw=45.0),
        "pitch",
        "tsr",
        "Ct",
        ax=axes[1, 1],
        levels=levels,
        cmap="plasma",
    )

    # Plot zero-yaw optimal
    dat = (
        df_trajectory.filter(yaw=0.0)
        .filter(pl.col("Cp") == pl.col("Cp").max())
        .select("pitch", "tsr")
    )

    for ax in axes.ravel():
        (strat1,) = ax.plot(
            dat["pitch"],
            dat["tsr"],
            "*",
            color="black",
            label=r"$\gamma=0^o$",
            ms=8,
            zorder=10,
        )

    # Plot 45 deg yaw optimal
    dat = (
        df_trajectory.filter(yaw=YAW2)
        .filter(pl.col("Cp") == pl.col("Cp").max())
        .select("pitch", "tsr")
    )
    for ax in axes[:, 1]:
        (strat2,) = ax.plot(
            dat["pitch"],
            dat["tsr"],
            "*",
            color="tab:orange",
            label=r"$\gamma=45^o$",
            ms=8,
            zorder=10,
        )

    # Plot zero-yaw trajectory
    for ax in axes.ravel():
        dat = df_trajectory.filter(yaw=0.0)
        (strat3,) = ax.plot(
            dat["pitch"],
            dat["tsr"],
            ls="-",
            color="black",
            label=r"$\gamma=0^o$",
            ms=8,
            zorder=10,
        )

    # Plot 45 degree yaw trajectory
    for ax in axes[:, 1]:
        dat = df_trajectory.filter(yaw=YAW2)
        (strat4,) = ax.plot(
            dat["pitch"],
            dat["tsr"],
            ls="-",
            color="tab:orange",
            label=r"$\gamma=45^o$",
            ms=8,
            zorder=10,
        )

    # Plot where the global optimal goes
    dat = df_trajectory.group_by("yaw", maintain_order=True).agg(
        pl.col("pitch").where(pl.col("Cp") == pl.col("Cp").max()).first(),
        pl.col("tsr").where(pl.col("Cp") == pl.col("Cp").max()).first(),
    )
    for ax in axes[:, 1]:
        ax.plot(dat["pitch"], dat["tsr"], "tab:orange", lw=1, ls="--")

    # Legend, including reordering so lines are at bottom
    axes[0, 0].legend(
        title=r"$C_{P,max}$",
        handles=[strat1, strat2],
        ncol=3,
        bbox_to_anchor=(0.2, 1.15),
        loc="lower left",
    )
    axes[0, 1].legend(
        title="Thrust-minimising derating",
        handles=[strat3, strat4],
        ncol=3,
        bbox_to_anchor=(0.2, 1.15),
        loc="lower left",
    )

    # Titles (as text)
    bbox_props = dict(boxstyle="round", ec="None", fc="white", alpha=0.7, mutation_aspect=0.2)
    text_props = dict(
        fontsize=12, color="k", ha="left", va="bottom", zorder=1000000000, bbox=bbox_props
    )
    axes[0, 0].text(
        0.02,
        1.01,
        r"a) $C_P$ ($\gamma=" + f"{0}" + "^o$)",
        transform=axes[0, 0].transAxes,
        **text_props,
    )
    axes[0, 1].text(
        0.02,
        1.01,
        r"b) $C_P$ ($\gamma=" + f"{YAW2}" + "^o$)",
        transform=axes[0, 1].transAxes,
        **text_props,
    )
    axes[1, 0].text(
        0.02,
        1.01,
        r"c) $C_T$ ($\gamma=" + f"{0}" + "^o$)",
        transform=axes[1, 0].transAxes,
        **text_props,
    )
    axes[1, 1].text(
        0.02,
        1.01,
        r"d) $C_T$ ($\gamma=" + f"{YAW2}" + "^o$)",
        transform=axes[1, 1].transAxes,
        **text_props,
    )

    # Add colorbar
    cbar = plt.colorbar(CF_Cp, cax=axes[0, 2])
    cbar.set_label(label=r"$C_P~$(-)")

    cbar = plt.colorbar(CF_Ct, cax=axes[1, 2], aspect=20)
    cbar.set_label(label=r"$C_T~$(-)")

    plt.savefig(utils.FIGDIRFORPAPER / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df_surface, df_trajectory = generate(regenerate=False)
    plot(df_surface, df_trajectory)


if __name__ == "__main__":
    main()
