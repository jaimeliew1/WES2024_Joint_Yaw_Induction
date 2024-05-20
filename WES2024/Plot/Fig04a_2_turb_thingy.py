"""
Figure 4:
2 turbine LES

I am not sure what this figure will do, but it should include results from the
LES case.

Key points:
- ??

"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from WES2024 import utils
from WES2024.Generate import JFM_layout_gridsearch_BEM

FILESTEM = Path(__file__).stem


# @utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False) -> pl.DataFrame:
    df = JFM_layout_gridsearch_BEM.generate(regenerate=regenerate)
    return df


def plot(df: pl.DataFrame):
    Ctprime_opt, yaw_opt, _ = (
        df.group_by("Ctprime_target", "yaw_target")
        .agg(pl.col("Cp").mean())
        .filter(Cp=pl.col("Cp").max())
        .rows()[0]
    )
    df_piv = df.pivot(
        index="Ctprime_target", columns="yaw_target", values="Cp", aggregate_function="mean"
    )
    Ctprime = df_piv["Ctprime_target"].to_numpy()
    yaw = np.array(df_piv.columns[1:], dtype=float)
    Cp = df_piv.to_numpy()[:, 1:]
    # Cp[Cp < 0.01] = 0.01
    # Cp[np.isnan(Cp)] = 0.02
    fig, ax = plt.subplots(1, 1)

    levels = np.arange(0.3, 0.601, 0.025)
    CF_Cp = ax.contourf(yaw, Ctprime, Cp, levels=levels, cmap="cividis")
    CS = ax.contour(yaw, Ctprime, Cp, levels=levels, colors="k", linewidths=0.8)
    ax.clabel(CS, inline=True, fontsize=10)

    # Add optimal
    ax.plot(yaw_opt, Ctprime_opt, "*y")

    # Add colorbar
    cbar = plt.colorbar(CF_Cp, ax=ax, aspect=20)
    cbar.set_label(label=r"$C_P~$(-)")

    # axis labels
    ax.set_xlabel(r"yaw (deg)")
    ax.set_ylabel(r"$C_T'$")

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=False)
    plot(df)


if __name__ == "__main__":
    main()
