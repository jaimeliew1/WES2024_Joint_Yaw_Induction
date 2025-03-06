"""
Plot farm-wide increase in power estimated from
MITWindfarm under different wind farm control. 

Kirby Heck
2025 January 24
"""

import numpy as np
import polars as pl
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

from WES2024.LES_new.step_3_optimize_controllers import LES_input_dir

LES_output_dir = Path(__file__).parent / "LES_output"

figpath = Path(__file__).parent / "figs"
figpath.mkdir(exist_ok=True, parents=True)


def run(
    fname="MITWindfarm_controller_gain.png",
    search="MITWindfarm*LESnew*.csv",
    inputdir=LES_input_dir,
):
    df_ls = []
    for f in inputdir.glob(search):
        print(f.name)
        df = pl.read_csv(f)
        df_ls.append(df)

    custom_order = {"nocontrol": 0, "thrustcontrol": 1, "yawcontrol": 2, "jointcontrol": 3}
    df_agg = pl.concat(df_ls).group_by("controller").mean()
    gain = df_agg["Cp"] / df_agg.filter(controller="nocontrol")["Cp"] - 1
    df_agg = df_agg.with_columns(
        [pl.Series("$C_P$ Gain", gain), pl.col("controller").map_dict(custom_order).alias("order")]
    ).sort(by="order")

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(df_agg, x="controller", y="$C_P$ Gain")
    ax.set_ylim([0, 0.12])
    plt.tight_layout()
    plt.savefig(figpath / fname, dpi=300)
    plt.close()


if __name__ == "__main__":
    run()
    run(
        fname="Old_controller_gain_TI.png",
        search="wdir-2.5_unifiedTI*.csv",
        inputdir=Path(__file__).parent.parent / "LES" / "LES_newinput",
    )
    run(
        fname="Old_controller_gain.png",
        search="wdir-2.5_unifiedTI_*.csv",
        inputdir=Path(__file__).parent.parent / "LES" / "LES_newinput",
    )

