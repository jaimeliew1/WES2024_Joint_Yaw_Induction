"""
Figure 2:
2 turbine wind direction sweep

This figure shows the optimal setpoints (yaw and Ct') using four different
control strategies (None, yaw, thrust, and joint control).

Key points:
- Joint control performs better than all other methods.
- Yaw control is discontinuous.
- Joint control is smooth.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from WES2024 import utils
from WES2024.Generate import two_turbine_AD

FILESTEM = Path(__file__).stem

REGENERATE = False


def generate(regenerate=False):
    df = two_turbine_AD.generate(regenerate=regenerate)
    return df


def plot(df: pl.DataFrame):
    fig, axes = plt.subplots(3, 1, sharex=True)

    for method in df["method"].unique():
        _df = (
            df.filter(pl.col("method") == method)
            .group_by("wdir")
            .agg(pl.col("Cp").mean(), pl.col("yaw").first(), pl.col("Ctprime").first())
        ).sort("wdir")

        _wdir = _df["wdir"].to_numpy()
        _Cp, _Ctprime, _yaw = (
            np.array(_df["Cp"].to_numpy()),
            np.array(_df["Ctprime"].to_numpy()),
            np.array(np.rad2deg(_df["yaw"].to_numpy())),
        )
        # Remove wdir=0 case for yaw control to show discontinuity
        if method == "YawControl":
            _Cp[np.abs(_wdir) < 1e-3] = np.nan
            _Ctprime[np.abs(_wdir) < 1e-3] = np.nan
            _yaw[np.abs(_wdir) < 1e-3] = np.nan

        axes[0].plot(_wdir, _Cp, label=method)
        axes[1].plot(_wdir, _Ctprime, label=method)
        axes[2].plot(_wdir, _yaw, label=method)

    axes[-1].set_xlabel("wind direction [deg]")

    axes[0].set_ylabel("$C_P$")
    axes[1].set_ylabel("$C_T'$")
    axes[2].set_ylabel(r"$\gamma$ [deg]")

    axes[0].set_ylim(0.3, 0.60)
    axes[1].set_ylim(1.0, 2.2)
    axes[2].set_ylim(-30.0, 30.0)
    axes[0].legend()

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=REGENERATE)
    plot(df)


if __name__ == "__main__":
    main()
