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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from WES2024 import utils
from WES2024.Generate import diamond_BEM

FILESTEM = Path(__file__).stem

REGENERATE = False


def generate(regenerate=False) -> pl.DataFrame:
    df = diamond_BEM.generate(regenerate=regenerate)
    return df


def plot(df: pl.DataFrame):

    df_piv = df.group_by("method").agg(pl.col("Cp").mean())
    Cp_ref = df_piv.filter(pl.col("method") == "NoControl")["Cp"]
    print(df_piv.with_columns(pl.col("Cp") / Cp_ref - 1))

    fig, ax = plt.subplots(1, 1, sharex=True)

    for method in df["method"].unique():
        _df = (
            df.filter(pl.col("method") == method).group_by("wdir").agg(pl.col("Cp").mean())
        ).sort("wdir")

        _wdir = _df["wdir"].to_numpy()
        _Cp = np.array(_df["Cp"].to_numpy())

        ax.plot(_wdir, _Cp, label=method)

    ax.set_xlabel("wind direction [deg]")

    ax.set_ylabel("$C_P$")

    ax.legend()

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=REGENERATE)
    plot(df)


if __name__ == "__main__":
    main()
