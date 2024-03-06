"""
Figure 11:
Diamond wind direction sweep BEM

This figure shows the optimal BEM setpoints (yaw, pitch, TSR, Ct' ) using four
different control strategies (None, yaw, thrust, and joint control).

Something is really slow when doing ThrustControlBEM. Look into this. Its fine.
2DOF: 2.2 seconds per optimisation
2DOF: 90 seconds per optimisation
3DOF: 200 seconds per optimisation

(it took 7 hours to run this)

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
layout = Square(6.0, 5).rotate(45)
wdirs = np.arange(0.0, 90.0, 0.05)



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


def plot(df: pl.DataFrame):

    df_piv = df.group_by("method").agg(pl.col("Cp").mean())
    Cp_ref = df_piv.filter(pl.col("method") == "NoControl")["Cp"]
    print(df_piv.with_columns(pl.col("Cp") / Cp_ref - 1))

    fig, ax = plt.subplots(1, 1, sharex=True)

    for method in methods:
        _df = (
            df.filter(pl.col("method") == method).group_by("wdir").agg(pl.col("Cp").mean())
        ).sort("wdir")

        _wdir = _df["wdir"].to_numpy()
        _Cp = np.array(_df["Cp"].to_numpy())

        ax.plot(_wdir, _Cp, label=method)

    ax.set_xlabel("wind direction [deg]")

    ax.set_ylabel("$C_P$")

    # ax.set_ylim(0.4, 0.51)
    ax.legend()

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=REGENERATE)
    plot(df)


if __name__ == "__main__":
    main()
