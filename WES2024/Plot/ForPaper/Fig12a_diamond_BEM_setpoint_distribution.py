"""
Figure 12:
Diamond BEM setpoint distribution

This figure shows how the optimal BEM setpoints are distributed for the Diamond
layout.

Key points:
- pitch and tsr follow minimum thrust trajectory (??)
- Some other nice geometric take aways??
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from WES2024 import utils
from WES2024.Generate import diamond_BEM, minCt_trajectory, pitch_tsr_surface
from WES2024.Plot.ForPaper.Fig00_single_turbine_surface import plot_surface

FILESTEM = Path(__file__).stem

REGENERATE = False

XLIM = (-4, 6)
YLIM = (6, 10.5)


def generate(regenerate=False) -> tuple[pl.DataFrame, ...]:
    df_diamond = diamond_BEM.generate(regenerate=regenerate)
    df_surface = pitch_tsr_surface.generate(regenerate=regenerate)
    df_trajectory = minCt_trajectory.generate(regenerate=regenerate)

    return df_diamond, df_surface, df_trajectory


def overlay_setpoint_scatter(df: pl.DataFrame, ax: plt.Axes, norm, cmap=None) -> any:

    graph = sns.scatterplot(
        df,
        x="pitch",
        y="tsr",
        hue="abs_yaw",
        hue_norm=norm,
        palette=cmap,
        legend=False,
        s=3,
        edgecolors=None,
        ax=ax,
    )

    return graph


def plot(df: pl.DataFrame, df_surface: pl.DataFrame, df_trajectory: pl.DataFrame):
    scatter_cmap = plt.cm.Greys_r  # sns.light_palette("seagreen", as_cmap=True)
    fig, _axes = plt.subplots(1, 5, width_ratios=[1, 0.1, 0.2, 1, 0.1], figsize=np.array((8, 3)))
    plt.subplots_adjust(wspace=0.1)

    cbar_axes = _axes[1], _axes[4]
    axes = _axes[0], _axes[3]
    _axes[2].set_axis_off()
    norm = mpl.colors.Normalize(0, 60)

    df = df.rename(dict(setpoint_0="pitch", setpoint_1="tsr")).with_columns(
        np.rad2deg(pl.col("yaw")),
        np.rad2deg(pl.col("pitch")),
        np.rad2deg(np.abs(pl.col("yaw"))).alias("abs_yaw"),
    )

    # Plot tsr-pitch surface
    df_surface = (
        df_surface.rename(dict(setpoint_0="pitch", setpoint_1="tsr"))
        .filter(pl.col("pitch").degrees().is_between(XLIM[0] - 0.1, XLIM[1] + 0.1))
        .filter(pl.col("tsr").is_between(YLIM[0] - 0.1, YLIM[1] + 0.1))
        .filter(pl.col("yaw") == 0)
        .with_columns(pl.col("pitch").degrees())
    )

    levels = np.arange(0, 0.60, 0.05)
    CF_Cp = plot_surface(df_surface, "pitch", "tsr", "Cp", ax=axes[0], levels=levels)
    levels = np.arange(0, 2, 0.1)
    CF_Ct = plot_surface(df_surface, "pitch", "tsr", "Ct", ax=axes[1], levels=levels, cmap="plasma")

    # Plot minimum thrust trajectories
    for yaw, _df in df_trajectory.filter(pl.col("yaw").is_in([0, 15, 30])).group_by(
        "yaw", maintain_order=True
    ):
        axes[0].plot(
            np.rad2deg(_df["pitch"]),
            _df["tsr"],
            "-",
            lw=1.5,
            c=scatter_cmap(norm(yaw)),
            label=r"$\gamma=" + f"{yaw}" + r"^o$",
        )
        axes[1].plot(
            np.rad2deg(_df["pitch"]),
            _df["tsr"],
            "-",
            lw=1.5,
            c=scatter_cmap(norm(yaw)),
            label=r"$\gamma=" + f"{yaw}" + r"^o$",
        )

    # Scatter set points
    overlay_setpoint_scatter(df, axes[0], norm, cmap=scatter_cmap)
    overlay_setpoint_scatter(df, axes[1], norm, cmap=scatter_cmap)

    cbar = plt.colorbar(CF_Cp, cax=cbar_axes[0])
    cbar.ax.set_title(r"$C_P~$(-)")
    cbar = plt.colorbar(CF_Ct, cax=cbar_axes[1])
    cbar.ax.set_title(r"$C_T~$(-)")

    axes[1].tick_params(labelleft=False)
    axes[1].set_ylabel(None)

    axes[0].set_xlabel(r"$\theta_p$ (deg)")
    axes[1].set_xlabel(r"$\theta_p$ (deg)")
    axes[0].set_ylabel(r"$\lambda$ (-)")
    axes[0].set_xlim(*XLIM)
    axes[0].set_ylim(*YLIM)

    axes[1].set_xlim(*XLIM)
    axes[1].set_ylim(*YLIM)

    axes[0].legend(
        title="Minimum thrust trajectory", loc="lower left", ncol=3, bbox_to_anchor=(0.0, 1.08)
    )

    plt.savefig(utils.FIGDIRFORPAPER / f"{FILESTEM}.png", dpi=500, bbox_inches="tight")


def main():
    df_quarter, df_surface, df_trajectory = generate(regenerate=REGENERATE)
    plot(df_quarter.filter(pl.col("method") == "JointControl"), df_surface, df_trajectory)


if __name__ == "__main__":
    main()
