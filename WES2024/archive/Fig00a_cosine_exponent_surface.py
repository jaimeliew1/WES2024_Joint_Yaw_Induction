from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from tqdm import tqdm
import foreach

from WES2024.Generate import pitch_tsr_surface, minCt_trajectory
from WES2024 import utils


FILESTEM = Path(__file__).stem
# Use Latex Fonts
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)

PITCHES = np.deg2rad(np.arange(-15, 15.001, 0.5))
TSRS = np.arange(5, 12.001, 0.125)

XLIM = (-5, 5)
YLIM = (5, 11.5)


def _fit_cos_exponent(yaws: np.typing.ArrayLike, cps: np.typing.ArrayLike) -> float:
    def error(p):
        return sum((np.cos(yaws) ** p - cps) ** 2)

    res = minimize_scalar(error, bounds=(0, 10))

    return res.x


def fit_cos_exponent(x):
    pitch, tsr, yaw, Cp, Ct = x
    Cp_ref = np.array(Cp)[np.array(yaw) == 0][0]
    Ct_ref = np.array(Ct)[np.array(yaw) == 0][0]

    exp_p = _fit_cos_exponent(np.deg2rad(yaw), np.array(Cp) / Cp_ref)
    exp_t = _fit_cos_exponent(np.deg2rad(yaw), np.array(Ct) / Ct_ref)

    return dict(pitch=pitch, tsr=tsr, exp_p=exp_p, exp_t=exp_t)


def process(df: pl.DataFrame) -> pl.DataFrame:
    # params = list(df.group_by(("pitch", "tsr")))
    params = list(df.group_by(("pitch", "tsr")).agg(pl.col("yaw", "Cp", "Ct")).iter_rows())

    out = foreach(fit_cos_exponent, params, parallel=False)
    # out = []
    # for (pitch, tsr), _df in tqdm(df.group_by(("pitch", "tsr"))):
    #     Cp_ref = _df.filter(pl.col("yaw") == 0.0)["Cp"][0]
    #     Ct_ref = _df.filter(pl.col("yaw") == 0.0)["Ct"][0]
    #     exp_p = fit_cos_exponent(np.deg2rad(_df["yaw"]), _df["Cp"] / Cp_ref)
    #     exp_t = fit_cos_exponent(np.deg2rad(_df["yaw"]), _df["Ct"] / Ct_ref)

    #     out.append(dict(pitch=pitch, tsr=tsr, exp_p=exp_p, exp_t=exp_t))

    return pl.from_dicts(out)


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
    df_trajectory = None
    # df_trajectory = (
    #     minCt_trajectory.generate(regenerate=regenerate)
    #     .with_columns(
    #         pl.col("pitch").degrees(),
    #     )
    #     .with_columns(type=pl.lit("trajectory"))
    # )

    # df_surface = df_surface.filter(pl.col("tsr") < 8, pl.col("tsr") > 7, pl.col("pitch").abs() < 1)
    df_surface = process(df_surface)
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
    df_piv_Cp = df.pivot(
        index=y, columns=x, values=z, aggregate_function=aggregate_function, maintain_order=True
    ).sort(y)
    # breakpoint()

    Y = df_piv_Cp[y].to_numpy()
    X = np.array(df_piv_Cp.columns[1:], dtype=float)
    Z = df_piv_Cp.to_numpy()[:, 1:]
    Z[Z < 0] = 0.01

    CF = ax.contourf(X, Y, Z, levels=levels, **kwargs)
    CS = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.8)
    ax.clabel(CS, inline=True, fontsize=10)

    return CF


def plot(df_surface: pl.DataFrame, df_trajectory: pl.DataFrame):

    df_surface = df_surface.filter(pl.col("exp_p") < 4)
    # fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(8, 4))
    fig, axes = plt.subplots(1, 4, width_ratios=[1, 0.1, 1, 0.1], figsize=np.array((8, 4)))

    # Set axes limits
    axes[0].set_xlim(*XLIM)
    axes[0].set_ylim(*YLIM)

    # filter data to fit axis limits
    df_surface = df_surface.filter(pl.col("tsr") <= YLIM[1] * 1.01)

    # Axis labels
    # [ax.set_xlabel(r"Pitch, $\theta_p~$(deg) ") for ax in axes[1, :2]]
    # [ax.set_ylabel(r"Tip Speed Ratio, $\lambda~$(-)") for ax in axes[:, 0]]

    # Plot surfaces
    levels = np.arange(0, 4, 0.1)
    CF_Cp = plot_surface(
        df_surface,
        "pitch",
        "tsr",
        "exp_p",
        ax=axes[0],
        # levels=levels,
        cmap="viridis",
    )

    # levels = np.arange(0, 2, 0.1)
    # CF_Ct = plot_surface(
    #     df_surface,
    #     "pitch",
    #     "tsr",
    #     "Ct",
    #     ax=axes[1],
    #     levels=levels,
    #     cmap="plasma",
    # )

    # Titles (as text)
    bbox_props = dict(boxstyle="round", ec="None", fc="white", alpha=0.7, mutation_aspect=0.2)
    text_props = dict(
        fontsize=12, color="k", ha="left", va="bottom", zorder=1000000000, bbox=bbox_props
    )
    # axes[0].text(
    #     0.02,
    #     1.01,
    #     r"a) $C_P$ ($\gamma=" + f"{0}" + "^o$)",
    #     transform=axes[0].transAxes,
    #     **text_props,
    # )
    # axes[1].text(
    #     0.02,
    #     1.01,
    #     r"b) $C_P$ ($\gamma=" + f"{YAW2}" + "^o$)",
    #     transform=axes[0, 1].transAxes,
    #     **text_props,
    # )

    # Add colorbar
    # cbar = plt.colorbar(CF_Cp, cax=axes[0, 2])
    # cbar.set_label(label=r"$C_P~$(-)")

    # cbar = plt.colorbar(CF_Ct, cax=axes[1, 2], aspect=20)
    # cbar.set_label(label=r"$C_T~$(-)")
    print(utils.FIGDIR / f"{FILESTEM}.png")
    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df_surface, df_trajectory = generate(regenerate=False)
    print(df_surface)
    plot(df_surface, df_trajectory)


if __name__ == "__main__":
    main()
