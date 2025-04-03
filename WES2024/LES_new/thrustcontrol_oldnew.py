"""
Should we re-run the thrust control cases? Take a look at 
how the C_T' setpoints differ. 

The version 1 data uses x0 = 3 in the calibration, while
the version 2 data uses x0 = 1 in the calibration. The net
result is that the C_T' set points are nearly identical, with
differences on the order of 1% and a maximum difference of 3%. 

Let's not re-run the thrust control case. 

Kirby Heck
2025 March 19
"""

import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
import WES2024

base = Path(__file__).parent

if __name__ == "__main__":

    path_0 = base / "../LES/LES_newinput/diamond_wdir-2.5_unified_thrustcontrol.csv"
    df_0 = pl.read_csv(path_0).with_columns(pl.lit(0).alias("version"))

    path_1 = base / "./LES_input/MITWindfarm_wdir-2.5_LESnew_thrustcontrol.csv"
    df_1 = pl.read_csv(path_1).with_columns(pl.lit(1).alias("version"))

    path_2 = base / "./LES_input/iter_01/MITWindfarm_wdir-2.5_LESnew_thrustcontrol.csv"
    df_2 = pl.read_csv(path_2).with_columns(pl.lit(2).alias("version"))

    df = pl.concat([df_1, df_2])

    df_diff = df_1.join(df_2, on="turbine", how="inner", suffix="_r").with_columns(
        (((pl.col("Ctprime") - pl.col("Ctprime_r")) / pl.col("Ctprime_r") * 100)).alias("Ctprime_diff")
    )

    fig, axs = plt.subplots(figsize=(8, 3), ncols=2)
    axs[0].scatter(df["turbine"], df["Ctprime"], c=df["version"])
    axs[1].scatter(df_diff["turbine"], df_diff["Ctprime_diff"])
    plt.subplots_adjust(wspace=0.3)
    for ax in axs: 
        ax.set_xlabel("Turbine no.")
    axs[0].set_ylabel("$C_T'$")
    axs[1].set_ylabel("Rel. diff. in $C_T'$ (\\%)")
    plt.show()

    # these setpoints are almost identical