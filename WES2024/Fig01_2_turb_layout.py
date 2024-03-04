"""
Figure 1:   
2 turbine layout

This figure qualitatively shows a yaw steering case using the 2 turbine layout.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from mitwindfarm import Plotting
from mitwindfarm.windfarm import Windfarm

from WES2024 import Fig02_2_turb_wdir_sweep_AD, utils

FILESTEM = Path(__file__).stem


def generate(regenerate=False) -> pl.DataFrame:
    df = Fig02_2_turb_wdir_sweep_AD.generate(regenerate)
    return df


def plot(df: pl.DataFrame):
    windfarm = Windfarm()

    _df = df.filter(pl.col("method") == "JointControl").filter(np.abs(pl.col("wdir") - 5) < 0.01)

    plt.figure()

    windfarm_sol = utils.from_polars(_df, windfarm)
    Plotting.plot_windfarm(windfarm_sol)
    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df = generate(regenerate=False)
    plot(df)


if __name__ == "__main__":
    main()
