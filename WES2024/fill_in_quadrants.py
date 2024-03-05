from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import utils

input_fn = Path("data/Fig07_diamond_wdir_sweep_many_spacings.csv")
FILESTEM = Path(__file__).stem




def plot_single(df: pl.DataFrame, layout: pl.DataFrame, ax):

    df = df.with_columns(np.rad2deg(pl.col("yaw")))

    for i, (x, y) in enumerate(layout.iter_rows()):
        _df = (
            df.filter(pl.col("turbine") == i)
            .select(np.deg2rad(pl.col("wdir")), pl.col("Ctprime"), pl.col("yaw"))
            .sort("wdir")
        )

        # plot turbine number
        ax.text(x, y, f"{i+1}", c="r", ha="center", va="center", fontsize=5)

        R = 0.75

        # rose_x = (_df["pitch"] + 10) * R * (-np.cos(_df["wdir"])) + x
        # rose_y = (_df["pitch"] + 10) * R * (np.sin(_df["wdir"])) + y
        # ax.plot(rose_x, rose_y, "r-", lw=1)

        rose_x = (_df["Ctprime"] + 0) * R * (-np.cos(_df["wdir"])) + x
        rose_y = (_df["Ctprime"] + 0) * R * (np.sin(_df["wdir"])) + y
        ax.plot(rose_x, rose_y, "g-", lw=0.5)

        R = 0.02
        rose_x = (_df["yaw"] + 90) * R * (-np.cos(_df["wdir"])) + x
        rose_y = (_df["yaw"] + 90) * R * (np.sin(_df["wdir"])) + y
        ax.plot(rose_x, rose_y, "b-", lw=0.5)
        ax.set(xticks=[], yticks=[])
        ax.axis("equal")





def plot(df: pl.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=np.array([8, 4]))

    layout = _df_to_plot = df.filter(pl.col("wdir") == 0.0).select("x", "y")
    plot_single(df, layout, axes[0])

    df_quarter = df.filter(pl.col("wdir") < 90)

    df_processed = utils.fill_in_other_quadrants(df_quarter)
    plot_single(df_processed, layout, axes[1])

    axes[0].set_title("full")
    axes[1].set_title("filled in")
    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=1000, bbox_inches="tight")


def main():
    # df = pl.read_csv(input_fn)
    df = (
        pl.scan_csv(input_fn)
        .filter(pl.col("method") == "JointControl")
        .filter(pl.col("min_dist") == 6.0)
        .collect()
    )
    plot(df)


if __name__ == "__main__":
    main()
