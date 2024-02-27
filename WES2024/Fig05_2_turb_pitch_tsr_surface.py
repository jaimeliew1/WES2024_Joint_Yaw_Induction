"""
Figure 5:
2 turbine pitch-TSR surface

This figure shows two contour plots: Cp and CT as a function of pitch and tsr.
Overlaid are the set point trajectories for different control strategies over
the wind direction sweep.

Key points:
- Optimal derating trajectory follows a minimum thrust trajectory

thoughts:
- The trajectory is pretty short. is this an error?
- perhaps this should be shown for the multiturbine wind farm case only.

"""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from foreach import foreach
from matplotlib.gridspec import GridSpec
from MITRotor.ReferenceTurbines import IEA15MW
from mitwindfarm.Layout import Layout
from mitwindfarm.Rotor import BEM
from mitwindfarm.windfarm import Windfarm

from WES2024 import Fig03_2_turb_wdir_sweep_BEM, utils
from WES2024.optimise import PITCH_OPT, TSR_OPT

PARALLEL = True
REGENERATE = False

XLIM = (-4, 3)
YLIM = (7, 10)

layout = Layout([0.0], [0.0])
FILESTEM = Path(__file__).stem

pitches = np.linspace(-6, 6, 150)
tsrs = np.linspace(5, 15, 150)
yaws = [0.0]

windfarm = Windfarm(rotor_model=BEM(IEA15MW()))


def _generate(x) -> pl.DataFrame:
    pitch, tsr, yaw = x
    setpoints = [(np.deg2rad(pitch), tsr, np.deg2rad(yaw))]
    sol = windfarm(layout, setpoints)
    df = utils.to_polars(sol)
    return df


@utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}_pitch_tsr_surface.csv")
def generate_pitch_tsr_surface(regenerate=False) -> pl.DataFrame:
    params = list(itertools.product(pitches, tsrs, yaws))

    df_list = foreach(_generate, params, parallel=PARALLEL)

    return pl.concat(df_list)


def generate(regenerate=False) -> pl.DataFrame:
    df_surface = generate_pitch_tsr_surface(regenerate)
    df_opt = Fig03_2_turb_wdir_sweep_BEM.generate(regenerate)
    return df_surface, df_opt


def plot(df_surface: pl.DataFrame, df_opt: pl.DataFrame):
    # Extract and reshape contour data points.
    df_surface = (
        df_surface.rename(dict(setpoint_0="pitch", setpoint_1="tsr"))
        .filter(pl.col("pitch").degrees().is_between(XLIM[0] - 0.1, XLIM[1] + 0.1))
        .filter(pl.col("tsr").is_between(YLIM[0] - 0.1, YLIM[1] + 0.1))
    )

    df_piv_Cp = df_surface.filter(pl.col("yaw") == 0).pivot(
        index="tsr", columns="pitch", values="Cp", aggregate_function=None
    )
    df_piv_Ct = df_surface.filter(pl.col("yaw") == 0).pivot(
        index="tsr", columns="pitch", values="Ct", aggregate_function=None
    )
    tsr = df_piv_Cp["tsr"].to_numpy()
    pitch = np.rad2deg(np.array(df_piv_Cp.columns[1:], dtype=float))
    Cp = df_piv_Cp.to_numpy()[:, 1:]
    Cp[Cp < 0.01] = 0.01
    Cp[np.isnan(Cp)] = 0.02
    Ct = df_piv_Ct.to_numpy()[:, 1:]
    Ct[np.isnan(Ct)] = 0.02
    Ct[Ct < 0.01] = 0.01

    # Extract optimal control points
    df_opt = df_opt.filter(pl.col("turbine") == 0).rename(
        dict(setpoint_0="pitch", setpoint_1="tsr")
    )

    plt.figure(figsize=np.array((8, 2)))
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)

    axes = [plt.subplot(gs[0])]
    axes.append(plt.subplot(gs[1], sharey=axes[0]))

    [ax.set_xlabel(r"Pitch, $\theta_p~$(deg) ") for ax in axes]
    [ax.set_ylabel(r"Tip Speed Ratio, $\lambda~$(-)") for ax in axes]

    plt.xlim(*XLIM)
    plt.ylim(*YLIM)

    ## Plot surfaces
    levels = np.arange(0, 0.60, 0.05)
    CF_Cp = axes[0].contourf(pitch, tsr, Cp, levels=levels, cmap="viridis")
    CS = axes[0].contour(pitch, tsr, Cp, levels=levels, colors="k", linewidths=0.8)
    axes[0].clabel(CS, inline=True, fontsize=10)

    levels = np.arange(0, 2, 0.1)
    CF_Ct = axes[1].contourf(pitch, tsr, Ct, levels=levels, cmap="plasma")
    CS = axes[1].contour(pitch, tsr, Ct, levels=levels, colors="k", linewidths=0.8)
    axes[1].clabel(CS, inline=True, fontsize=10)

    # Plot optimal setpoints
    [ax.plot(np.rad2deg(PITCH_OPT), TSR_OPT, "*", label=r"$C_{p,max}$") for ax in axes]
    for method in ["ThrustControl", "JointControl"]:
        _df = df_opt.filter(pl.col("method") == method).sort("wdir")
        for ax in axes:
            ax.plot(np.rad2deg(_df["pitch"]), _df["tsr"], label=method)

    # Add colorbar
    cbar = plt.colorbar(CF_Cp, ax=axes[0], aspect=20)
    cbar.set_label(label=r"$C_P~$(-)")

    cbar = plt.colorbar(CF_Ct, ax=axes[1], aspect=20)
    cbar.set_label(label=r"$C_T~$(-)")

    # Add caption letters
    axes[0].text(0.0, 1.02, "a)", ha="left", va="bottom", transform=axes[0].transAxes)
    axes[1].text(0.0, 1.02, "b)", ha="left", va="bottom", transform=axes[1].transAxes)

    axes[0].legend(
        ncol=3,
        bbox_to_anchor=(1.2, 1.1),
        loc="lower center",
    )
    # Save figure to file
    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=500, bbox_inches="tight")
    plt.close()


def main():
    df_surface, df_opt = generate(regenerate=REGENERATE)
    plot(df_surface, df_opt)


if __name__ == "__main__":
    main()
