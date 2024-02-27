"""
Figure 11:
Diamond wind direction sweep BEM

This figure shows the optimal BEM setpoints (yaw, pitch, TSR, Ct' ) using four
different control strategies (None, yaw, thrust, and joint control).

Something is really slow when doing ThrustControlBEM. Look into this. Its fine.
2DOF: 2.2 seconds per optimisation
2DOF: 90 seconds per optimisation
3DOF: 200 seconds per optimisation

Key points:
- ??

"""
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from foreach import foreach
from MITRotor.ReferenceTurbines import IEA15MW
from mitwindfarm.Layout import Square
from mitwindfarm.Rotor import BEM
from mitwindfarm.windfarm import Windfarm
from optimise import JointControlBEM, NoControlBEM, ThrustControlBEM, YawControlBEM
from test_BEM_gradients import DualBEM

from WES2024 import utils

FILESTEM = Path(__file__).stem

REGENERATE = True
PARALLEL = True


windfarm = Windfarm(rotor_model=BEM(IEA15MW(), BEM_model=DualBEM))
layout = Square(10.0, 5).rotate(45)
wdirs = np.arange(0.0, 45.0, 0.05)


methods = {
    "NoControl": NoControlBEM,
    "YawControl": YawControlBEM,
    "ThrustControl": ThrustControlBEM,
    "JointControl": JointControlBEM,
}


def _generate(x):
    method, wdir = x
    sol = methods[method](layout.rotate(wdir), windfarm).optimise(
        Cp_constraint=None,
        use_gradients=True,
        verbose=False,
    )

    return utils.to_polars(sol).with_columns(
        pl.lit(method).alias("method"), pl.lit(wdir).alias("wdir")
    )


@utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False):
    params = list(product(methods, wdirs))

    dfs = foreach(_generate, params, parallel=PARALLEL)
    df = pl.concat(dfs)
    return df


def plot(df):
    fig, axes = plt.subplots(5, 1, sharex=True)

    for method in methods:
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

    axes[0].set_ylim(0.4, 0.51)
    axes[0].legend()

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=REGENERATE)
    plot(df)


if __name__ == "__main__":
    main()
