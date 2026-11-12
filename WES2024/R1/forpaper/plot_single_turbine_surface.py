# dualitic is imported first to ensure correct monkey patching.
import dualitic

"""
R1 paper figure: Cp and Ct surfaces over (pitch, TSR) at yaw = 0 and 45 deg, with the Ct'-minimising thrust-derating
trajectory overlaid. Same figure as WES2024/Plot/ForPaper/Fig00_single_turbine_surface.py, but data comes from the
R1 annulus-closure BEM LUT (WES2024.R1.generate.generate_single_turbine_surface, wakemodels.get_bem_lut_rotor)
instead of a live per-point BEM solve.

Usage: python -m WES2024.R1.forpaper.plot_single_turbine_surface
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from WES2024.R1 import figuresettings as fs
from WES2024.R1.generate import generate_single_turbine_surface as gen

YAW2 = 45  # deg
XLIM = (-15, 15)
YLIM = (5, 10.5)


def plot_surface(df: pl.DataFrame, x: str, y: str, z: str, ax: plt.Axes, levels=None, **kwargs):
    df_piv = df.pivot(index=y, columns=x, values=z)
    Y = df_piv[y].to_numpy()
    X = np.array(df_piv.columns[1:], dtype=float)
    Z = df_piv.to_numpy()[:, 1:]
    Z[Z < 0] = 0.01
    CF = ax.contourf(X, Y, Z, levels=levels, **kwargs)
    CS = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.8)
    ax.clabel(CS, inline=True, fontsize=7)
    return CF


def plot(df_surface: pl.DataFrame, df_trajectory: pl.DataFrame):
    fig, axes = plt.subplots(2, 3, width_ratios=[1, 1, 0.08], figsize=(6.5, 4))

    for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.sharex(axes[0, 0])
        ax.sharey(axes[0, 0])

    axes[0, 0].tick_params(labelbottom=False)
    axes[0, 1].tick_params(labelbottom=False)
    axes[0, 1].tick_params(labelleft=False)
    axes[1, 1].tick_params(labelleft=False)

    axes[0, 0].set_xlim(*XLIM)
    axes[0, 0].set_ylim(*YLIM)

    df_surface = df_surface.filter(pl.col("tsr") <= YLIM[1] * 1.01)

    [ax.set_xlabel(r"Pitch, $\theta_p$ (deg)") for ax in axes[1, :2]]
    [ax.set_ylabel(r"TSR, $\lambda$") for ax in axes[:, 0]]

    levels = np.arange(0, 0.51, 0.05)
    CF_Cp = plot_surface(df_surface.filter(yaw=0.0), "pitch", "tsr", "Cp", ax=axes[0, 0], levels=levels, cmap="viridis")
    plot_surface(df_surface.filter(yaw=YAW2), "pitch", "tsr", "Cp", ax=axes[0, 1], levels=levels, cmap="viridis")

    levels = np.arange(0, 1.51, 0.1)
    CF_Ct = plot_surface(df_surface.filter(yaw=0.0), "pitch", "tsr", "Ct", ax=axes[1, 0], levels=levels, cmap="plasma")
    plot_surface(df_surface.filter(yaw=YAW2), "pitch", "tsr", "Ct", ax=axes[1, 1], levels=levels, cmap="plasma")

    # zero-yaw optimum (max Cp on the trajectory == the unconstrained optimum)
    dat = df_trajectory.filter(yaw=0.0).filter(pl.col("Cp") == pl.col("Cp").max()).select("pitch", "tsr")
    for ax in axes[:, :2].ravel():
        (strat1,) = ax.plot(dat["pitch"], dat["tsr"], "*", color="black", label=r"$\gamma=0^\circ$", ms=8, zorder=10)

    # 45 deg yaw optimum
    dat = df_trajectory.filter(yaw=YAW2).filter(pl.col("Cp") == pl.col("Cp").max()).select("pitch", "tsr")
    for ax in axes[:, 1]:
        (strat2,) = ax.plot(dat["pitch"], dat["tsr"], "*", color="tab:orange", label=rf"$\gamma={YAW2}^\circ$", ms=8, zorder=10)

    # zero-yaw derate trajectory
    dat = df_trajectory.filter(yaw=0.0)
    for ax in axes[:, :2].ravel():
        (strat3,) = ax.plot(dat["pitch"], dat["tsr"], ls="-", color="black", label=r"$\gamma=0^\circ$", zorder=10)

    # 45 deg yaw derate trajectory
    dat = df_trajectory.filter(yaw=YAW2)
    for ax in axes[:, 1]:
        (strat4,) = ax.plot(dat["pitch"], dat["tsr"], ls="-", color="tab:orange", label=rf"$\gamma={YAW2}^\circ$", zorder=10)

    font_props = dict(fontsize=8, title_fontsize=8, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.13))
    axes[0, 0].legend(title=r"$C_{P,\mathrm{max}}$", handles=[strat1, strat2], **font_props)
    axes[0, 1].legend(title="Thrust-minimizing derating", handles=[strat3, strat4], **font_props)

    # bbox_props = dict(boxstyle="round", ec="None", fc="white", alpha=0.7)
    text_props = dict(fontsize=9, color="k", ha="left", va="bottom")
    axes[0, 0].text(0.0, 1.02, r"(a) $C_P$ ($\gamma=0^\circ$)", transform=axes[0, 0].transAxes, **text_props)
    axes[0, 1].text(0.0, 1.02, rf"(b) $C_P$ ($\gamma={YAW2}^\circ$)", transform=axes[0, 1].transAxes, **text_props)
    axes[1, 0].text(0.0, 1.02, r"(c) $C_T$ ($\gamma=0^\circ$)", transform=axes[1, 0].transAxes, **text_props)
    axes[1, 1].text(0.0, 1.02, rf"(d) $C_T$ ($\gamma={YAW2}^\circ$)", transform=axes[1, 1].transAxes, **text_props)

    cbar = plt.colorbar(CF_Cp, cax=axes[0, 2])
    cbar.set_label(r"$C_P$")
    cbar = plt.colorbar(CF_Ct, cax=axes[1, 2])
    cbar.set_label(r"$C_T$")

    fig.tight_layout()
    fs.save()


if __name__ == "__main__":
    df_surface, df_trajectory = gen.generate()
    plot(df_surface, df_trajectory)
