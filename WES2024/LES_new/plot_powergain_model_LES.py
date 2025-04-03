"""
Plot farm-wide increase in power estimated from
MITWindfarm under different wind farm control. 

Kirby Heck
2025 January 24
"""

import polars as pl
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

from WES2024.LES_new.step_3_optimize_controllers import LES_input_dir
from WES2024.LES_new.final_calibration import LES_output_dir

figpath = Path(__file__).parent / "figs"
figpath.mkdir(exist_ok=True, parents=True)
controller_labels = {
    "nocontrol": "No Control",
    "thrustcontrol": "Thrust Control",
    "yawcontrol": "Yaw Control",
    "jointcontrol": "Joint Control",
}

def plot_powergain(df, fname):
    print("Plotting power gain: ", fname)
    custom_order = {"nocontrol": 0, "thrustcontrol": 1, "yawcontrol": 2, "jointcontrol": 3}
    # df_agg = df.group_by(["controller", "simulator"]).mean()
    norm_values = (
        df.filter(pl.col("controller") == "nocontrol")
        .group_by("simulator")
        .mean()
        .select(["simulator", "Cp"])
        .rename({"Cp": "Cp_norm"})
    )
    df = df.join(norm_values, on="simulator")

    df = df.with_columns(
        (pl.col("Cp") / pl.col("Cp_norm") - 1).alias("$C_P$ Gain"),
        pl.col("controller").replace(custom_order, default=None).alias("order"),
        pl.col("controller").replace(controller_labels).alias("Controller"),
        pl.col("simulator").alias("Simulator"),  # just capitalize it!
    ).sort(by=["order", "simulator"])

    fig, ax = plt.subplots(figsize=(4.5, 3))
    sns.barplot(
        df,
        x="Controller",
        y="$C_P$ Gain",
        hue="Simulator",
        palette=["0.4", "tab:blue", ],
        errorbar=("sd", 2),
        edgecolor='k',
        lw=0.5,
        err_kws=dict(linewidth=1, color="k", ),
        capsize=0.2,
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim([-0.055, 0.225])
    plt.tight_layout()
    print("Saving figure", figpath / f"LES_model_Cp_gain_{fname}.png")
    plt.savefig(figpath / f"LES_model_Cp_gain_{fname}.png", dpi=300)
    plt.close()


def run(fsearch="LESnew"):
    df_ls = []

    # try to read farm-aggregated Cp
    try:
        farm_data = next(LES_output_dir.glob(f"*{fsearch}_farm_Cp*"))
        print("Reading: ", farm_data)
        df_ls.append(pl.read_csv(farm_data).with_columns(pl.col("farm_cp").alias("Cp")))
        read_LES_case = False

    except StopIteration as e:
        # no farm data exported
        read_LES_case = True

    for f in LES_input_dir.glob(f"*{fsearch}*.csv"):
        casename = f.stem.split("MITWindfarm_")[-1]
        print("Reading:", f)

        df_model = pl.read_csv(f).group_by(["controller", "simulator"]).mean()
        df_ls.append(df_model)
        if read_LES_case:
            f_LES = LES_output_dir / f"LES_{casename}.csv"
            df_LES = pl.read_csv(f_LES).mean().with_columns(
                pl.lit("LES").alias("simulator"), 
                pl.lit(casename.split("_")[-1]).alias("controller")
            )
            df_ls.append(df_LES)

    plot_powergain(pl.concat(df_ls, how="diagonal_relaxed"), fname=fsearch)


if __name__ == "__main__":
    run()
    # run(fsearch="unifiedTI")
