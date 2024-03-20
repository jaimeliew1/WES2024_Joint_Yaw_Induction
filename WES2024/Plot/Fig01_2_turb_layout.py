"""
Figure 1:   
2 turbine layout

This figure qualitatively shows a yaw steering case using the 2 turbine layout.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mitwindfarm import Plotting
from mitwindfarm.windfarm import Windfarm, WindfarmSolution

from WES2024 import utils
from WES2024.Generate import two_turbine_AD

FILESTEM = Path(__file__).stem


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

    xlim = (np.min(sol.layout.x) - pad, np.max(sol.layout.x) + 5)
    ylim = (np.min(sol.layout.y) - pad, np.max(sol.layout.y) + pad)

    _x, _y = np.linspace(*xlim, 1000), np.linspace(*ylim, 1000)
    xmesh, ymesh, x0, y0 = rotated_meshgrid(_x, _y, angle_rad)
    wsp = sol.windfield.wsp(xmesh, ymesh, np.zeros_like(xmesh))
    ax.imshow(wsp, extent=[*xlim, *ylim], vmin=-0.1, vmax=2.1, origin="lower", cmap="RdYlBu_r")

    for (turb_x, turb_y, _), rotor in zip(sol.layout, sol.rotors):
        # Draw turbine
        R = 0.5
        p = np.array([[0, 0], [+R, -R]])
        rotmat = np.array(
            [
                [np.cos(rotor.yaw - angle_rad), -np.sin(rotor.yaw - angle_rad)],
                [np.sin(rotor.yaw - angle_rad), np.cos(rotor.yaw - angle_rad)],
            ]
        )
        _turb_x = (turb_x - x0) * np.cos(-angle_rad) - (turb_y - y0) * np.sin(-angle_rad)
        _turb_y = (turb_x - x0) * np.sin(-angle_rad) + (turb_y - y0) * np.cos(-angle_rad)
        points = rotmat @ p + np.array([[_turb_x + x0], [_turb_y + y0]])
        # breakpoint()
        # color_denom = 0.65
        color = "k"  # plt.cm.plasma(rotor.Cp / color_denom)
        ax.plot(points[0, :], points[1, :], lw=2, c=color)
    
    # centerline of 2 turbines
    # plt.axhline(points[1, :].mean(), ls="--", c="k", lw=1)


    ax.set(frame_on=frame)
    if axis is False:
        ax.set(xticks=[], yticks=[])


def plot(df: pl.DataFrame):
    windfarm = Windfarm()

    _df = df.filter(pl.col("method") == "JointControl").filter(np.abs(pl.col("wdir") - 5) < 0.01)
    wdir = _df["wdir"][0]
    plt.figure()
    windfarm_sol = utils.from_polars(_df, windfarm)
    plot_windfarm(windfarm_sol, plt.gca(), angle=wdir, pad=2)

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=1000, bbox_inches="tight")
    plt.close()


def main():
    df = generate(regenerate=False)
    plot(df)


if __name__ == "__main__":
    main()
