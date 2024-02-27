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
import polars as pl

from WES2024 import utils


FILESTEM = Path(__file__).stem


# @utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False) -> pl.DataFrame:
    df = pl.DataFrame()
    return df


def plot(df: pl.DataFrame):
    plt.figure()

    # Placeholder text
    plt.annotate(
        f"INSERT {FILESTEM} HERE",
        xy=(0.5, 0.5),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=14,
        color="gray",
    )

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=False)
    plot(df)


if __name__ == "__main__":
    main()
