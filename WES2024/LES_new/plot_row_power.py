"""
Plot row power for each simulator, controller

Kirby Heck
2025 January 29
"""

import polars as pl
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from WES2024.LES_new.step_3_optimize_controllers import LES_input_dir
from WES2024.LES_new.compare_superposition import LES_pnormfact

from UnifiedMomentumModel import Momentum
unified = Momentum.UnifiedMomentum()
Betz = unified(2., 0)

LES_output_dir = Path(__file__).parent / "LES_output"

figpath = Path(__file__).parent / "figs"
figpath.mkdir(exist_ok=True, parents=True)

ROW_MAPPING = {
    **dict.fromkeys([0, 5, 10, 15, 20, 21, 22, 23, 24], 0),
    **dict.fromkeys([1, 6, 11, 16, 17, 18, 19], 1),
    **dict.fromkeys([2, 7, 12, 13, 14], 1),
    **dict.fromkeys([3, 9], 3),
    **dict.fromkeys([4], 4),
}

PLOT_ORDER = {"nocontrol": 0, "thrustcontrol": 1, "yawcontrol": 2, "jointcontrol": 3}


def plot_row(df, ax=None, hue="simulator", palette=None, ):
    if ax is None:
        _, ax = plt.subplots()

    palette = palette or ["0.4", "tab:blue"]

    sns.barplot(
        df,
        ax=ax,
        x="No. upstream",
        y="Cp",
        hue=hue,
        palette=palette,
        errorbar=("pi", 100),
        edgecolor='k', 
        lw=0.5,
        err_kws=dict(color="k", lw=0.5, ls='--'),
        capsize=0.3,
    )
    ax.set_ylabel("")
    return ax


def run(fsearch="LESnew"):
    df_ls = []

    for f in LES_input_dir.glob(f"*{fsearch}*.csv"):
        casename = f.stem.split("MITWindfarm_")[-1]
        print("Reading:", casename)

        # read model data:
        df_model = pl.read_csv(f).with_columns(
            pl.col("Cp") / Betz.Cp
        )
        df_ls.append(df_model)

        # now read LES data:
        f_LES = LES_output_dir / f"LES_{casename}.csv"
        df_LES = pl.read_csv(f_LES).with_columns(
            pl.lit("LES").alias("simulator"),
            pl.lit(casename.split("_")[-1]).alias("controller"),
            pl.col("turbine_id").alias("turbine"),
            pl.col("Cp") / LES_pnormfact 
        )
        df_ls.append(df_LES)

    df = (
        pl.concat(df_ls, how="diagonal_relaxed")
        .with_columns(
            pl.col("turbine").replace(ROW_MAPPING, default=None).alias("No. upstream"),
            pl.col("controller").replace(PLOT_ORDER, default=None).alias("order"),
        )
        .sort(by=["order", "simulator"])
    )

    ctrls = df["controller"].unique(maintain_order=True)
    fig, axs = plt.subplots(
        ncols=len(ctrls), figsize=(len(ctrls) * 1.5, 2.5), sharex=True, sharey=True
    )

    for ax, controller in zip(axs, ctrls):
        ax.set_title(controller)
        plot_row(df.filter(controller=controller), ax=ax)
        if ax != axs[1]: 
            ax.legend_.remove()

    plt.subplots_adjust(bottom=0.4)
    axs[1].legend(bbox_to_anchor=(1.1, -0.3), loc="upper center", title="Simulator", ncols=2)
    axs[0].set_ylabel("$P/P_1$")
    # axs[0].set_ylabel("$C_P$")
    for ax in axs:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    plt.savefig(figpath / "P_by_row_norm.png", dpi=300)
    plt.close()
    print("Done")


if __name__ == "__main__":
    run()
