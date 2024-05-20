"""
Figure 4:
2 turbine LES

I am not sure what this figure will do, but it should include results from the
LES case.

Key points:
- ??

"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import norm

from WES2024 import utils
from WES2024.Generate import diamond_BEM_uncertainty

FILESTEM = Path(__file__).stem


stds = [1, 2, 3, 4, 5]
# @utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False) -> pl.DataFrame:
    df = diamond_BEM_uncertainty.generate(regenerate=regenerate)
    return df

KEY = "Cp"
x_refined = np.linspace(-15, 15, 100)

def plot(df: pl.DataFrame):
    plt.figure()
    # df = df.filter(pl.col("method") == "JointControl")
    for params, _df in df.sort("wdir_offset").group_by("wdir", "turbine"):

        for method, __df in _df.group_by("method"):
            # y_refined = np.interp(x_refined, __df["wdir_offset"], __df[KEY])


            # plt.plot(x_refined, y_refined, ".", **utils.line_params[method])
            plt.plot(__df["wdir_offset"], __df[KEY], ".", **utils.line_params[method])

        for std in stds:
            pdf=norm(0, std).pdf(x_refined)
            plt.plot(x_refined, pdf, ":")
            # Cp_bar = np.trapz(pdf * _df[KEY], _df["wdir_offset"])
            # plt.plot(_df["wdir_offset"], pdf * _df[KEY], "--")
            # plt.plot(0, Cp_bar, ".")
            
        # break
        plt.title(f"{params}")
        plt.legend()
        plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")
        plt.close()
        breakpoint()


def main():
    df = generate(regenerate=False)
    plot(df)


if __name__ == "__main__":
    main()
