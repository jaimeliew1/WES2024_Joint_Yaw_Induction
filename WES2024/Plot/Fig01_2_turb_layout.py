"""
Figure 1:   
2 turbine layout

This figure qualitatively shows a yaw steering case using the 2 turbine layout.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mitwindfarm import Plotting
from mitwindfarm.windfarm import Windfarm

from WES2024 import utils
from WES2024.Generate import two_turbine_AD

FILESTEM = Path(__file__).stem


def generate(regenerate=False) -> pl.DataFrame:
    df = two_turbine_AD.generate(regenerate=regenerate)
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
