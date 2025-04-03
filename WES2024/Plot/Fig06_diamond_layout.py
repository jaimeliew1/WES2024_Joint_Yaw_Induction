"""
Figure 6:   
diamond wind farm layout

This figure qualitatively shows a yaw steering case using the 25 turbine diamond
layout.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from mitwindfarm import Plotting, Windfarm

from WES2024 import utils
from WES2024.Generate import diamond_AD

FILESTEM = Path(__file__).stem


def generate(regenerate=False):
    df = diamond_AD.generate(regenerate=regenerate)
    return df


def plot(df: pl.DataFrame):
    windfarm = Windfarm()

    _df = (
        df.filter(pl.col("method") == "NoControl")
        .filter(pl.col("wdir") == 2.5)
        .filter(pl.col("min_dist") == 6.0)
    )
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
