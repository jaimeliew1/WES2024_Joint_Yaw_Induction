from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from mitwindfarm import Plotting
from mitwindfarm.windfarm import Windfarm

from WES2024 import utils, Fig02_2_turb_wdir_sweep_AD


FILESTEM = Path(__file__).stem


def generate(regenerate=False) -> pl.DataFrame:
    df = Fig02_2_turb_wdir_sweep_AD.generate(regenerate)
    return df


def plot(df: pl.DataFrame):
    windfarm = Windfarm()

    _df = df.filter(pl.col("method") == "JointControl").filter(pl.col("wdir") == 5.0)
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
