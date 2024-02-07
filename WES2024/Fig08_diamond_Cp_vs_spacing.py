from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from WES2024 import utils, Fig07_diamond_wdir_sweep_many_spacings

FILESTEM = Path(__file__).stem


def generate(regenerate=False) -> pl.DataFrame:
    df = Fig07_diamond_wdir_sweep_many_spacings.generate(regenerate)
    return df


def plot(df: pl.DataFrame):
    df_farm_Cp = df.pivot(columns="method", index="min_dist", values="Cp", aggregate_function="mean")
    df_farm_Cp = df_farm_Cp.select(
        pl.col("min_dist"),
        pl.exclude("min_dist", "NoControl") / pl.col("NoControl") * 100 - 100,
    )
    print(df_farm_Cp)

    methods = df_farm_Cp.columns
    methods.remove("min_dist")

    plt.figure()
    for method, _plot_params in Fig07_diamond_wdir_sweep_many_spacings.plot_params.items():
        if method == "NoControl":
            continue
        plt.plot(df_farm_Cp["min_dist"], df_farm_Cp[method], **_plot_params)

    plt.legend()

    plt.xlabel("Turbine spacing [D]")
    plt.ylabel(r"Power increase [\%]")

    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=False)
    plot(df)


if __name__ == "__main__":
    main()
