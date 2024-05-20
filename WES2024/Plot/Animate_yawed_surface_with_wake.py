from itertools import repeat
from pathlib import Path

import ffmpeg
import foreach
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mitwindfarm import Layout, Windfarm

from WES2024 import utils
from WES2024.Generate import minCt_trajectory, pitch_tsr_surface

TEMPDIR = Path("TEMP")
TEMPDIR.mkdir(exist_ok=True, parents=True)


windfarm = Windfarm()
layout = Layout(np.array([0.0]), np.array([0.0]))


def generate(regenerate=False) -> tuple[pl.DataFrame, ...]:
    df_surface = (
        pitch_tsr_surface.generate(regenerate=regenerate)
        .select(
            pl.col("setpoint_0").degrees().alias("pitch"),
            pl.col("setpoint_1").alias("tsr"),
            pl.col("yaw").degrees(),
            pl.col("Cp"),
            pl.col("Ct"),
        )
        .filter(pl.col("tsr") < 11)
    )

    df_traj = (
        minCt_trajectory.generate(regenerate=regenerate)
        .with_columns(pl.col("pitch").degrees())
        .filter(pl.col("yaw") != 22.5)
    )

    return df_surface, df_traj


def interpolate(
    value: float, on_key: str, sort_by: list[str], df_all: pl.DataFrame
) -> pl.DataFrame:
    val_lower = df_all.filter(pl.col(on_key) <= value)[on_key].max()
    val_upper = df_all.filter(pl.col(on_key) > value)[on_key].min()

    weight = (value - val_lower) / (val_upper - val_lower)

    df_lower = df_all.filter(pl.col(on_key) == val_lower).sort(by=sort_by)
    df_upper = df_all.filter(pl.col(on_key) == val_upper).sort(by=sort_by)

    df_interp = pl.DataFrame((1 - weight) * df_lower + weight * df_upper, schema=df_upper.columns)

    return df_interp


def _plot_surface(
    df: pl.DataFrame,
    x: str,
    y: str,
    z: str,
    ax: plt.Axes,
    levels=None,
    aggregate_function=None,
    cbar_label="",
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
    # ax.clabel(CS, inline=True, fontsize=10)

    # Colorbar
    cbar = plt.colorbar(CF, ax=ax, aspect=20)
    cbar.set_label(label=cbar_label)

    return CF


def _plot_turbine_wake(ax, yaw: float):
    sol = windfarm(layout, [(2.0, np.deg2rad(yaw))])
    x, y = np.linspace(-2, 10, 500), np.linspace(-3, 3, 500)
    xmesh, ymesh = np.meshgrid(x, y)

    wsp = sol.windfield.wsp(xmesh, ymesh, 0.0).T
    centerline_x = sol.wakes[0].x_centerline
    centerline_y = sol.wakes[0].centerline(centerline_x)

    ax.imshow(
        wsp,
        cmap="YlGnBu_r",
        extent=[y.min(), y.max(), x.min(), x.max()],
        vmin=0,
        vmax=1,
        origin="lower",
    )

    # Draw turbine
    turb_x, turb_y, yaw = 0, 0, np.deg2rad(yaw)
    R = 0.5
    p = np.array([[turb_x, turb_x], [turb_y + R, turb_y - R]])
    rotmat = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

    p = rotmat @ p

    ax.plot(p[1, :], p[0, :], "k", lw=5)

    # Plot centerline
    ax.plot(centerline_y, centerline_x, ":", c="tab:blue", lw=1)

    # Add yaw angle text
    ax.set_title(r"$\gamma=" + f"{np.rad2deg(yaw):2.0f}" + r"^o$")

    # # set axis limits
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 8)

    # remote ticks
    ax.set_xticks([])
    ax.set_yticks([])


def plot_single(
    df: pl.DataFrame, df_traj: pl.DataFrame, df_traj_all: pl.DataFrame, yaw: float, index: int
):
    # Set up axes
    fig, axes = plt.subplot_mosaic(
        [["wake", "Cp_contour"], ["wake", "Ct_contour"]],
        figsize=1.2 * np.array((6, 4)),
        gridspec_kw=dict(hspace=0.4, wspace=0.5),
    )

    # limits
    axes["Cp_contour"].set_ylim(5, 11)
    axes["Ct_contour"].set_ylim(5, 11)
    # Axis labels
    axes["Cp_contour"].set_xlabel(r"Pitch, $\theta_p~$(deg) ")
    axes["Ct_contour"].set_xlabel(r"Pitch, $\theta_p~$(deg) ")
    axes["Cp_contour"].set_ylabel(r"Tip Speed Ratio, $\lambda~$(-)")
    axes["Ct_contour"].set_ylabel(r"Tip Speed Ratio, $\lambda~$(-)")
    # Plot surfaces
    levels = np.arange(0, 0.60, 0.05)
    _plot_surface(
        df,
        "pitch",
        "tsr",
        "Cp",
        ax=axes["Cp_contour"],
        levels=levels,
        cmap="viridis",
        cbar_label="$C_P (-)$",
    )

    levels = np.arange(0, 2, 0.1)
    _plot_surface(
        df,
        "pitch",
        "tsr",
        "Ct",
        ax=axes["Ct_contour"],
        levels=levels,
        cmap="plasma",
        cbar_label="$C_T (-)$",
    )

    # Plot zero-yaw optima
    pitch, tsr = (
        df_traj_all.filter(yaw=0).filter(pl.col("derate") == 1).select("pitch", "tsr").rows()[0]
    )
    axes["Cp_contour"].plot(pitch, tsr, "*", color="black", label=r"$\gamma=0^o$", ms=8)
    axes["Ct_contour"].plot(pitch, tsr, "*", color="black", label=r"$\gamma=0^o$", ms=8)

    # Plot yawed optima
    pitch, tsr = df_traj.filter(pl.col("derate") == 1).select("pitch", "tsr").rows()[0]
    axes["Cp_contour"].plot(pitch, tsr, "*", color="tab:orange", label=r"$\gamma=0^o$", ms=8)
    axes["Ct_contour"].plot(pitch, tsr, "*", color="tab:orange", label=r"$\gamma=0^o$", ms=8)

    # Plot zero-yaw trajectory
    axes["Cp_contour"].plot(
        df_traj_all.filter(yaw=0)["pitch"],
        df_traj_all.filter(yaw=0)["tsr"],
        ls="-",
        color="black",
        label=r"$\gamma=0^o$",
        ms=8,
    )
    axes["Ct_contour"].plot(
        df_traj_all.filter(yaw=0)["pitch"],
        df_traj_all.filter(yaw=0)["tsr"],
        ls="-",
        color="black",
        label=r"$\gamma=0^o$",
        ms=8,
    )

    # Plot where the global optima goes
    dat = (
        df_traj_all.filter(pl.col("yaw") <= yaw)
        .group_by("yaw", maintain_order=True)
        .agg(
            pl.col("pitch").where(pl.col("Cp") == pl.col("Cp").max()).first(),
            pl.col("tsr").where(pl.col("Cp") == pl.col("Cp").max()).first(),
        )
    )
    axes["Cp_contour"].plot(dat["pitch"], dat["tsr"], "tab:orange", lw=1, ls="--")
    axes["Ct_contour"].plot(dat["pitch"], dat["tsr"], "tab:orange", lw=1, ls="--")

    # Plot yawed trajectory
    axes["Cp_contour"].plot(
        df_traj["pitch"], df_traj["tsr"], ls="-", color="tab:orange", label=r"$\gamma=0^o$", ms=8
    )
    axes["Ct_contour"].plot(
        df_traj["pitch"], df_traj["tsr"], ls="-", color="tab:orange", label=r"$\gamma=0^o$", ms=8
    )

    # Plot yawed turbine
    _plot_turbine_wake(axes["wake"], yaw)

    plt.savefig(TEMPDIR / f"pitch_tsr_contour_{index:03}.png", dpi=300, bbox_inches="tight")
    plt.close()


def animate(dir_to_animate: Path, out_fn: Path, framerate: int = 20, wildcard: str = "/*.png"):
    dir_to_animate = Path(dir_to_animate)
    N_files = len(list(dir_to_animate.iterdir()))

    print(f"animating {N_files} frames...")

    (
        ffmpeg.input(
            dir_to_animate.as_posix() + wildcard,
            pattern_type="glob",
            framerate=framerate,
            pix_fmt="yuv420p",
        )
        .output(out_fn.as_posix())
        .run(overwrite_output=True, quiet=True)
    )


def smoothstep(x):
    # Ensure x is in the range [0, 1]
    x = np.maximum(0, np.minimum(1, x))

    # Smoothstep function
    return x * x * (3 - 2 * x)


def _func(x):
    i, (yaw, df_surface, df_traj) = x
    _df_surface = interpolate(yaw, on_key="yaw", sort_by=["pitch", "tsr"], df_all=df_surface)
    _df_traj = interpolate(yaw, on_key="yaw", sort_by=["pitch", "tsr"], df_all=df_traj)
    plot_single(_df_surface, _df_traj, df_traj, yaw, i)


def main(fps: int):
    df_surface, df_traj = generate(regenerate=False)
    t_max = 10
    dt = 1 / fps
    t = np.arange(0, t_max, dt)
    tstart1, tend1 = 1, 4
    tstart2, tend2 = 6, 9
    yaws = (
        45
        * smoothstep((t - tstart1) / (tend1 - tstart1))
        * (1 - smoothstep((t - tstart2) / (tend2 - tstart2)))
    )

    params = list(enumerate(zip(yaws, repeat(df_surface), repeat(df_traj))))
    foreach(_func, params, context="spawn", parallel=True, processes=16)

    animate(TEMPDIR, utils.FIGDIR / "pitch_tsr_animation.mp4", framerate=fps)


if __name__ == "__main__":
    main(fps=20)
