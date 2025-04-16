"""
Figure 8:   
Diamond layout - power increase due to control versus turbine spacing

This figure shows how much power is gained by performing optimal wind farm
control (versus not performing control) versus turbine spacing. Assumes uniform
wind rose.

Key points:
- Yaw control outperforms thrust control (known in literature).
- Joint control outperforms yaw control (novel result).
- Benefit of control is larger for for close spacings.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from WES2024 import utils
from WES2024.Generate import diamond_AD

FILESTEM = Path(__file__).stem


def generate(regenerate=False) -> pl.DataFrame:
    df = diamond_AD.generate(regenerate=regenerate)
    return df


def plot(df: pl.DataFrame):
    df_farm_Cp = df.pivot(
        columns="method", index="min_dist", values="Cp", aggregate_function="mean"
    ).sort("min_dist")
    df_farm_Cp = df_farm_Cp.select(
        pl.col("min_dist"),
        pl.exclude("min_dist", "NoControl") / pl.col("NoControl") * 100 - 100,
    )
    print(df_farm_Cp)

    methods = df_farm_Cp.columns
    methods.remove("min_dist")

    plt.figure(figsize=(4,3))
    for method, _plot_params in utils.line_params.items():
        if method == "NoControl":
            continue
        plt.plot(df_farm_Cp["min_dist"], df_farm_Cp[method], **_plot_params)

    plt.legend()

    plt.xlabel("Turbine spacing [D]")
    plt.ylabel(r"Windfarm power increase [\%]")

    plt.xlim(2, 10)
    plt.ylim(0, 9)

    plt.savefig(utils.FIGDIRFORPAPER / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=False)
    plot(df)


if __name__ == "__main__":
    main()
