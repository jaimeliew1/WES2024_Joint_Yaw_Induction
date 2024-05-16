from pathlib import Path
from typing import Callable

import ffmpeg
from foreach import foreach
import matplotlib.pyplot as plt
import numpy as np
from mitwindfarm.Layout import Layout
from mitwindfarm.windfarm import Windfarm, WindfarmSolution
from rich import print
from PIL import Image

from WES2024 import utils

TEMPDIR = Path("TEMP2")
TEMPDIR.mkdir(exist_ok=True, parents=True)

YAW_MAX = 30
CTPRIME_MIN = 1.0
windfarm = Windfarm()
layout = Layout(np.array([0.0, 8.0]), np.array([0.0, 0.5]))
P_farm_ref = windfarm(layout, [(2.0, 0.0), (2.0, 0.0)]).Cp


def yaw_text(sol: WindfarmSolution) -> str:
    return r"$\gamma=" + f"{np.rad2deg(sol.rotors[0].yaw):2.0f}" + r"^o$"


def Ct_text(sol: WindfarmSolution) -> str:
    # return r"$C_T=" + f"{sol.rotors[0].Ct:2.2f}$"
    return f"Thrust: {sol.rotors[0].Ct/(8/9)*100:2.0f}\%"


def _plot_turbine_wake(
    ax,
    yaw: float,
    Ctprime: float,
    title: str,
    P_farm_ref: float,
    setpoint_text_func: Callable[[WindfarmSolution], str],
):
    sol = windfarm(layout, [(Ctprime, np.deg2rad(yaw)), (2.0, 0.0)])
    x, y = np.linspace(-2, 10, 500), np.linspace(-3, 3, 500)
    xmesh, ymesh = np.meshgrid(x, y)

    wsp = sol.windfield.wsp(xmesh, ymesh, 0.0).T
    centerline_x = sol.wakes[0].x_centerline
    centerline_y = sol.wakes[0].centerline(centerline_x)

    ax.imshow(
        wsp.T,
        cmap="YlGnBu_r",
        extent=[x.min(), x.max(), y.min(), y.max()],
        vmin=0,
        vmax=1,
        origin="lower",
    )

    # Draw turbines
    for ((turb_x, turb_y, _), rotor) in zip(sol.layout, sol.rotors):
        yaw = -rotor.yaw
        R = 0.5
        p = np.array(
            [
                [turb_y + R, turb_y - R],
                [turb_x, turb_x],
            ]
        )
        rotmat = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

        p = rotmat @ p

        ax.plot(p[1, :], p[0, :], "k", lw=5 * rotor.Ctprime / 2.0)

    # Plot centerline
    ax.plot(centerline_x, centerline_y, ":", c="tab:blue", lw=1)

    # Add title
    ax.set_title(title)

    # Add set point text
    setpoint_text = setpoint_text_func(sol)
    ax.text(sol.layout.x[0], sol.layout.y[0] + R + 0.1, setpoint_text, va="bottom", ha="center")

    # Add power increase text
    ax.text(
        0.5,
        0.01,
        "Farm power: " + f"{100*(sol.Cp/P_farm_ref - 1):+2.1f}\%",
        va="bottom",
        ha="center",
        transform=ax.transAxes,
    )
    # # set axis limits
    ax.set_ylim(-2, 2)
    ax.set_xlim(-2, 10)

    # remote ticks
    ax.set_xticks([])
    ax.set_yticks([])


def plot_single(x: tuple[int, float]):
    index, weight = x

    # Set up axes
    fig, axes = plt.subplot_mosaic(
        [["yaw"], ["derating"]],
        figsize=1.2 * np.array((4, 3)),
        gridspec_kw=dict(hspace=0.4, wspace=0.5),
    )
    yaw = weight * YAW_MAX
    # Plot yawed turbine
    _plot_turbine_wake(
        axes["yaw"],
        yaw,
        2.0,
        "Yaw control",
        P_farm_ref,
        yaw_text,
    )

    Ctprime = 2.0 - weight * CTPRIME_MIN
    _plot_turbine_wake(
        axes["derating"],
        0.0,
        Ctprime,
        "Induction control",
        P_farm_ref,
        Ct_text,
    )

    plt.savefig(
        TEMPDIR / f"yaw_and_induction_control_demo_{index:03}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.11,
    )
    plt.close()

    resize_to_even_dimensions(
        TEMPDIR / f"yaw_and_induction_control_demo_{index:03}.png",
        TEMPDIR / f"yaw_and_induction_control_demo_{index:03}.png",
    )


def resize_to_even_dimensions(image_path, output_path):
    # Open an image file
    with Image.open(image_path) as img:
        # Get current dimensions
        width, height = img.size

        # Calculate new dimensions to be even
        new_width = width if width % 2 == 0 else width - 1
        new_height = height if height % 2 == 0 else height - 1

        # Resize the image
        resized_img = img.resize((new_width, new_height))

        # Save the resized image
        resized_img.save(output_path)


def animate_mp4(dir_to_animate: Path, out_fn: Path, framerate: int = 20, wildcard: str = "/*.png"):
    dir_to_animate = Path(dir_to_animate)
    N_files = len(list(dir_to_animate.iterdir()))

    print(f"animating {N_files} frames...")

    (
        ffmpeg.input(
            dir_to_animate.as_posix() + wildcard,
            pattern_type="glob",
            framerate=framerate,
        )
        .output(
            out_fn.as_posix(),
            pix_fmt="yuv420p",
        )
        .run(overwrite_output=True, quiet=False)
    )


def smoothstep(x):
    # Ensure x is in the range [0, 1]
    x = np.maximum(0, np.minimum(1, x))

    # Smoothstep function
    return x**2 * (3 - 2 * x)


def main(fps: int):
    t_max = 8
    dt = 1 / fps
    t = np.arange(0, t_max, dt)
    tstart1, tend1 = 0, 3
    tstart2, tend2 = 5, 8
    weight = smoothstep((t - tstart1) / (tend1 - tstart1)) * (
        1 - smoothstep((t - tstart2) / (tend2 - tstart2))
    )

    params = list(enumerate(weight))
    foreach(plot_single, params, context="spawn", parallel=True, processes=16)

    animate_mp4(TEMPDIR, utils.FIGDIR / "yaw_and_induction_control_demo.mp4", framerate=fps)


if __name__ == "__main__":
    main(fps=24)
