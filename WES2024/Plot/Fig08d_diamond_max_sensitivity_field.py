"""
Figure 1:   
2 turbine layout

This figure qualitatively shows a yaw steering case using the 2 turbine layout.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from dualitic import DualNumber
from mitwindfarm import Windfarm, WindfarmSolution

from WES2024 import utils
from WES2024.Generate import diamond_AD_sensitivity

FILESTEM = Path(__file__).stem

windfarm = Windfarm()
VLIM = 0.4


def generate(regenerate=False) -> pl.DataFrame:
    df = diamond_AD_sensitivity.generate(regenerate=regenerate)
    # df = utils.fill_in_other_quadrants(df)
    return df.filter(pl.col("min_dist") == 6.0).filter(pl.col("method") == "JointControl")


def rotated_meshgrid(x, y, angle_rad):
    # Something is wrong when used with a dual number angle.
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


def plot_windfarm(
    sol: WindfarmSolution,
    sol_grad: WindfarmSolution,
    ax=None,
    pad=1,
    frame=True,
    axis=False,
    angle=0,
    dual=False,
    vmin=None,
    vmax=None,
    cmap=None,
):
    if ax is None:
        _, ax = plt.subplots()

    angle_rad = np.deg2rad(angle)

    xlim = (np.min(sol.layout.x) - pad, np.max(sol.layout.x) + 5)
    ylim = (np.min(sol.layout.y) - pad, np.max(sol.layout.y) + pad)

    _x, _y = np.linspace(*xlim, 1000), np.linspace(*ylim, 1000)
    xmesh, ymesh, x0, y0 = rotated_meshgrid(_x, _y, angle_rad)
    wsp = sol_grad.windfield.wsp(xmesh, ymesh, np.zeros_like(xmesh))

    if dual:
        field = wsp.dual[:, :, 0]
    else:
        field = wsp.real

    ax.imshow(
        field,
        extent=[*xlim, *ylim],
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        cmap=cmap,
    )

    for (turb_x, turb_y, _), rotor in zip(sol.layout, sol.rotors):
        # Draw turbine
        R = 0.5
        py = np.array([+R, -R])

        _turb_x = (turb_x - x0) * np.cos(-angle_rad) - (turb_y - y0) * np.sin(-angle_rad) + x0
        _turb_y = (turb_x - x0) * np.sin(-angle_rad) + (turb_y - y0) * np.cos(-angle_rad) + y0

        points_x = -np.sin(rotor.yaw - angle_rad) * py + _turb_x
        points_y = np.cos(rotor.yaw - angle_rad) * py + _turb_y

        color = "k"
        # ax.plot(_turb_x, _turb_y, ".k")
        # ax.plot(_turb_x.real, _turb_y.real, ".k")
        ax.plot(points_x, points_y, lw=2, c=color)

    ax.set(frame_on=frame)
    if axis is False:
        ax.set(xticks=[], yticks=[])


def plot(df: pl.DataFrame, wdir: float):
    _df = df.filter(pl.col("wdir") == wdir)
    windfarm_sol = utils.from_polars(_df, windfarm)

    rotation = DualNumber([0.0], [[1.0]])
    layout = windfarm_sol.layout.rotate(rotation)

    sol_grad = windfarm(layout, windfarm_sol.setpoints)
    fig, axes = plt.subplots(1, 2, figsize=np.array([8, 4]))
    plot_windfarm(
        windfarm_sol,
        sol_grad,
        axes[0],
        angle=0,
        pad=2,
        dual=False,
        vmin=-0.1,
        vmax=2.1,
        cmap="RdYlBu_r",
    )
    plot_windfarm(
        windfarm_sol,
        sol_grad,
        axes[1],
        angle=0,
        pad=2,
        dual=True,
        vmin=-VLIM,
        vmax=VLIM,
        cmap="PiYG",
    )

    axes[0].set_title(r"$u(x, y, z)/u_\infty$")
    axes[1].set_title(r"$\frac{\partial u(x, y, z)}{\partial \alpha}/u_\infty$")
    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=1000, bbox_inches="tight")
    plt.close()


def main():
    df = generate(regenerate=False)
    most_intersting_wdir = (
        df.group_by("wdir")
        .agg(pl.col("dCpdwdir").mean().abs())
        .filter(dCpdwdir=pl.col("dCpdwdir").max())["wdir"][0]
    )
    print(f"The most intersting wind direction is {most_intersting_wdir} degrees.")

    plot(df, wdir=most_intersting_wdir)


if __name__ == "__main__":
    main()
