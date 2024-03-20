"""
Figure 1:   
2 turbine layout

This figure qualitatively shows a yaw steering case using the 2 turbine layout.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mitwindfarm.windfarm import Windfarm, WindfarmSolution

from WES2024 import utils
from WES2024.Generate import two_turbine_AD
from dualitic import DualNumber

from rich import print

FILESTEM = Path(__file__).stem

windfarm = Windfarm()


def generate(regenerate=False) -> pl.DataFrame:
    df = two_turbine_AD.generate(regenerate=regenerate)
    return df


def rotated_meshgrid(x, y, angle_rad):
    xmesh, ymesh = np.meshgrid(x, y)
    x0, y0 = xmesh.mean(), ymesh.mean()
    xmesh -= x0
    ymesh -= y0
    # Convert angle to radians
    angle_radians = angle_rad

    # Rotate the coordinates
    x_rotated = xmesh * np.cos(angle_radians) - ymesh * np.sin(angle_radians)
    y_rotated = xmesh * np.sin(angle_radians) + ymesh * np.cos(angle_radians)

    return x_rotated + x0, y_rotated + y0, x0, y0


def plot_windfarm(sol: WindfarmSolution, ax=None, pad=1, frame=True, axis=False, angle=0):
    if ax is None:
        _, ax = plt.subplots()

    angle_rad = np.deg2rad(angle)

    xlim = (np.min(sol.layout.x.real) - pad, np.max(sol.layout.x.real) + 5)
    ylim = (np.min(sol.layout.y.real) - pad, np.max(sol.layout.y.real) + pad)

    _x, _y = np.linspace(*xlim, 1000), np.linspace(*ylim, 1000)
    xmesh, ymesh, x0, y0 = rotated_meshgrid(_x, _y, angle_rad)
    wsp = sol.windfield.wsp(xmesh, ymesh, np.zeros_like(xmesh))
    ax.imshow(
        wsp.dual[:, :, 0],
        extent=[*xlim, *ylim],
        # vmin=-0.1,
        # vmax=2.1,
        origin="lower",
        cmap="RdYlBu_r",
    )

    for (turb_x, turb_y, _), rotor in zip(sol.layout, sol.rotors):
        # Draw turbine
        R = 0.5
        py = np.array([+R, -R])

        _turb_x = (turb_x - x0) * np.cos(-angle_rad) - (turb_y - y0) * np.sin(-angle_rad)
        _turb_y = (turb_x - x0) * np.sin(-angle_rad) + (turb_y - y0) * np.cos(-angle_rad)

        points_x = -np.sin(rotor.yaw - angle_rad) * py + _turb_x + x0
        points_y = np.cos(rotor.yaw - angle_rad) * py + _turb_y + y0

        color = "k"
        ax.plot(points_x.real, points_y.real, lw=2, c=color)

    ax.set(frame_on=frame)
    if axis is False:
        ax.set(xticks=[], yticks=[])


def plot_sensitivity_field(df: pl.DataFrame):

    _df = df.filter(pl.col("method") == "JointControl").filter(np.abs(pl.col("wdir") - 5) < 0.01)
    wdir = _df["wdir"][0]
    plt.figure()
    windfarm_sol = utils.from_polars(_df, windfarm)

    wdir = DualNumber([0.0], [[1.0]])
    layout = windfarm_sol.layout.rotate(wdir)

    sol_grad = windfarm(layout, windfarm_sol.setpoints)
    plot_windfarm(sol_grad, plt.gca(), angle=wdir, pad=2)

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=1000, bbox_inches="tight")
    plt.close()


def plot_sensitivity_vs_wdir(df: pl.DataFrame):
    print(df.columns)
    out = []
    for (method, wdir), _df in df.group_by("method", "wdir"):
        windfarm_sol = utils.from_polars(_df, windfarm)
        rotate = DualNumber([0.0], [[1.0]])
        layout = windfarm_sol.layout.rotate(rotate)

        sol_grad = windfarm(layout, windfarm_sol.setpoints)
        Cp = sol_grad.Cp
        out.append(dict(method=method, wdir=wdir, Cp=Cp.real[0], sensitivity=Cp.dual[0, 0]))

    out = pl.from_dicts(out)

    fig, axes = plt.subplots(2, 1, sharex=True)

    for method, _df in out.group_by("method"):
        _df = _df.sort("wdir")
        axes[0].plot(_df["wdir"], _df["Cp"], c=utils.controller_colors[method])
        axes[1].plot(_df["wdir"], _df["sensitivity"], c=utils.controller_colors[method])
        wdir_max_sensitivity = _df.filter(sensitivity=pl.col("sensitivity").max())["wdir"][0]
        axes[0].axvline(wdir_max_sensitivity, c="k", ls="--", lw=1)
        axes[1].axvline(wdir_max_sensitivity, c="k", ls="--", lw=1)

    plt.savefig(utils.FIGDIR / f"{FILESTEM}_vs_wdir.png", dpi=1000, bbox_inches="tight")
    plt.close()


def plot(df: pl.DataFrame):
    plot_sensitivity_vs_wdir(df)
    plot_sensitivity_field(df)


def main():
    df = generate(regenerate=False)
    plot(df)


if __name__ == "__main__":
    main()
