from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from WES2024 import utils, Fig07_diamond_wdir_sweep_many_spacings


FILESTEM = Path(__file__).stem


def generate(regenerate=False) -> pl.DataFrame:
    df = Fig07_diamond_wdir_sweep_many_spacings.generate(regenerate)
    return df


def plot(df: pl.DataFrame):
    plt.figure()
    _df = df.filter(pl.col("min_dist") ==6).filter(pl.col("method") == "YawControl")
    _df2 = df.filter(pl.col("min_dist") ==6).filter(pl.col("method") == "JointControl")
    print(f"{_df["yaw"].std()=}")
    print(f"{_df2["yaw"].std()=}")
    print(f"{_df["yaw"].min()=}")
    print(f"{_df2["yaw"].min()=}")
    print(f"{_df["yaw"].max()=}")
    print(f"{_df2["yaw"].max()=}")
    # Create a histogram
    plt.hist(_df["yaw"], bins=30, color='skyblue', edgecolor='black')
    plt.hist(_df2["yaw"], bins=30, edgecolor='black')
    # Add labels and title
    plt.title('Histogram Example')
    plt.xlabel('Values')
    plt.ylabel('Frequency')


    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=False)
    plot(df)


if __name__ == "__main__":
    main()
