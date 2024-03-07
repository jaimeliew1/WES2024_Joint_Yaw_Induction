"""
Figure 3:
2 turbine wind direction sweep BEM

This figure shows the optimal BEM setpoints (yaw, pitch, TSR, Ct' ) using four
different control strategies (None, yaw, thrust, and joint control).

Key points:
- Joint control performs better than all other methods.
- Yaw control is discontinuous.
- Joint control is smooth.
- Yaw control has non-constant thrust. i.e. yaw and thrust are coupled quantities.

"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from WES2024 import utils
from WES2024.Generate import two_turbine_BEM

REGENERATE = False
FILESTEM = Path(__file__).stem


def generate(regenerate=False):
    df = two_turbine_BEM.generate(regenerate=regenerate)
    return df


def plot(df):
    fig, axes = plt.subplots(5, 1, sharex=True)

    for method in df["method"].unique():
        _df = (
            df.filter(pl.col("method") == method)
            .group_by("wdir")
            .agg(
                pl.col("Cp").mean(),
                pl.col("setpoint_0").where(pl.col("turbine") == 0).first(),
                pl.col("setpoint_1").where(pl.col("turbine") == 0).first(),
                pl.col("setpoint_2").where(pl.col("turbine") == 0).first(),
                pl.col("Ctprime").where(pl.col("turbine") == 0).first(),
            )
        ).sort("wdir")

        _wdir = _df["wdir"].to_numpy()
        _Cp, _pitch, _tsr, _yaw, _Ctprime = (
            np.array(_df["Cp"].to_numpy()),
            np.array(np.rad2deg(_df["setpoint_0"].to_numpy())),
            np.array(_df["setpoint_1"].to_numpy()),
            np.array(np.rad2deg(_df["setpoint_2"].to_numpy())),
            np.array(_df["Ctprime"].to_numpy()),
        )
        # Remove wdir=0 case for yaw control to show discontinuity
        if method == "YawControl":
            _Cp[np.abs(_wdir) < 1e-3] = np.nan
            _pitch[np.abs(_wdir) < 1e-3] = np.nan
            _tsr[np.abs(_wdir) < 1e-3] = np.nan
            _yaw[np.abs(_wdir) < 1e-3] = np.nan
            _Ctprime[np.abs(_wdir) < 1e-3] = np.nan

        axes[0].plot(_wdir, _Cp, label=method)
        axes[1].plot(_wdir, _pitch, label=method)
        axes[2].plot(_wdir, _tsr, label=method)
        axes[3].plot(_wdir, _yaw, label=method)
        axes[4].plot(_wdir, _Ctprime, label=method)

    axes[-1].set_xlabel("wind direction [deg]")

    axes[0].set_ylabel("$C_P$")
    axes[1].set_ylabel(r"$\theta_p$ [deg]")
    axes[2].set_ylabel(r"$\lambda$")
    axes[3].set_ylabel(r"$\gamma$ [deg]")
    axes[4].set_ylabel(r"$C_T'$ [deg]")

    axes[0].set_ylim(0.3, 0.51)
    axes[3].set_ylim(-30.0, 30.0)
    axes[4].set_ylim(1.0, 2.2)
    axes[0].legend()

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=REGENERATE)
    plot(df)


if __name__ == "__main__":
    main()
