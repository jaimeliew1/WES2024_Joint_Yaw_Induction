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

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
import numpy as np
import polars as pl
from MITRotor.ReferenceTurbines import IEA15MW
from mitwindfarm.Rotor import BEM
import MITRotor
from mitwindfarm.windfarm import Windfarm
from WES2024.Generate import diamond_BEM, minCt_trajectory, pitch_tsr_surface
from WES2024.BEM_gradients import DualBEM

from WES2024 import  utils

FILESTEM = Path(__file__).stem

REGENERATE = False

XLIM = (-3, 6)
YLIM = (5, 10)
# YAWS = np.arange(0.0, 50.1, 20.0)
# windfarm = Windfarm(rotor_model=BEM(IEA15MW(), BEM_model=DualBEM))


# @utils.cache_polars(utils.CACHEDIR / "min_Ct_trajectory.csv")
# def generate_min_Ct_trajectory(regenerate=False) -> pl.DataFrame:
#     rotor = MITRotor.ReferenceTurbines.IEA10MW()
#     bem = MITRotor.BEM.BEM(rotor=rotor)
#     out = []
#     for yaw in YAWS:
#         out.append(utils.generate_derate_strat(bem, np.deg2rad(yaw)).with_columns(yaw=yaw))
#     out = pl.concat(out)

#     return out


def generate(regenerate=False) -> tuple[pl.DataFrame, ...]:
    df_diamond = diamond_BEM.generate(regenerate=regenerate)
    df_surface = pitch_tsr_surface.generate(regenerate=regenerate)
    df_trajectory = minCt_trajectory.generate(regenerate=regenerate)

    return df_diamond, df_surface, df_trajectory


def plot(df: pl.DataFrame, df_surface: pl.DataFrame, df_trajectory: pl.DataFrame):
    fig = plt.figure()
    ax = plt.gca()
    norm = mpl.colors.Normalize(0, 50)

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

    ## Plot surfaces
    levels = np.arange(0, 0.60, 0.05)
    CF_Cp = ax.contourf(pitch, tsr, Cp, levels=levels, cmap="viridis")
    CS = ax.contour(pitch, tsr, Cp, levels=levels, colors="k", linewidths=0.8)
    ax.clabel(CS, inline=True, fontsize=10)

    # Plot minimum thrust trajectories
    for yaw, _df in df_trajectory.group_by("yaw", maintain_order=True):
        plt.plot(
            np.rad2deg(_df["pitch"]),
            _df["tsr"],
            "-",
            lw=1.5,
            c=plt.cm.magma(norm(yaw)),
            label=r"$\gamma=" + f"{yaw}" + r"^o$",
        )

    # Scatter set points

    graph = sns.scatterplot(
        df.to_pandas(),
        x="pitch",
        y="tsr",
        hue="abs_yaw",
        hue_norm=norm,
        palette="magma",
        legend=False,
        s=3,
        edgecolors=None,
        ax=plt.gca(),
    )
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="magma"),
        ax=plt.gca(),
        orientation="vertical",
        label=r"$|\gamma|$ [deg]",
    )
    graph.set_xlabel(r"$\theta_p$ [deg]")
    graph.set_ylabel(r"$\lambda$ [-]")

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)

    ax.legend(title="Minimum thrust\ntrajectory")

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df_quarter, df_surface, df_trajectory = generate(regenerate=REGENERATE)
    plot(df_quarter.filter(pl.col("method") == "JointControl"), df_surface, df_trajectory)


if __name__ == "__main__":
    main()
