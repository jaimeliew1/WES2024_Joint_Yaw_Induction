from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from mitwindfarm import Plotting
from mitwindfarm.windfarm import Windfarm

from WES2024 import utils, Fig07_diamond_wdir_sweep_many_spacings


FILESTEM = Path(__file__).stem


def generate(regenerate=False) -> pl.DataFrame:
    df = Fig07_diamond_wdir_sweep_many_spacings.generate(regenerate)
    return df


def plot(df: pl.DataFrame):
    windfarm = Windfarm()

    _df = df.filter(pl.col("method") == "JointControl").filter(pl.col("wdir") == 5.0).filter(pl.col("min_dist") == 6.0)
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
