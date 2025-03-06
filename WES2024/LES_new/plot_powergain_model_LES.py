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

LES_output_dir = Path(__file__).parent / "LES_output"

figpath = Path(__file__).parent / "figs"
figpath.mkdir(exist_ok=True, parents=True)


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
    ).sort(by=["order", "simulator"])

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(
        df,
        x="controller",
        y="$C_P$ Gain",
        hue="simulator",
        palette=["0.4", "tab:blue", ],
        errorbar=("sd", 2),
        edgecolor='k',
        lw=0.5,
        err_kws=dict(linewidth=1, color="k", ),
        capsize=0.2,
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim([-0.055, 0.205])
    plt.tight_layout()
    print("Saving figure", figpath / f"LES_model_Cp_gain_{fname}.png")
    plt.savefig(figpath / f"LES_model_Cp_gain_{fname}.png", dpi=300)
    plt.close()


def run(fsearch="LESnew"):
    df_ls = []

    # try to read farm-aggregated Cp
    try:
        farm_data = next(LES_output_dir.glob(f"{fsearch}_farm_Cp*"))
        df_ls.append(pl.read_csv(farm_data).with_columns(pl.col("farm_cp").alias("Cp")))
        read_LES_case = False

    except StopIteration as e:
        # no farm data exported
        read_LES_case = True

    for f in LES_input_dir.glob(f"*{fsearch}*.csv"):
        casename = f.stem.split("MITWindfarm_")[-1]
        print("Reading:", casename)

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


def run_old(fsearch="*LESnew*.csv"):
    df_ls = []

    for f in LES_output_dir.glob(fsearch):
        casename = f.name.split("LES_")[-1]
        print("Running:", casename)
        try:
            model_inputs = next(LES_input_dir.glob(f"*{casename}"))
        except StopIteration as e:
            print("\tNo model inputs found")
            continue

        df2 = pl.read_csv(model_inputs)  # model
        df1 = pl.read_csv(f)
        df1 = df1.with_columns(
            pl.lit("LES").alias("simulator"),
            pl.col("turbine_id").alias("turbine"),
            pl.lit(df2["controller"].unique()).alias("controller"),
        )  # .drop("turbine_id", "Cp_rms")

        df2 = df2.select([col for col in df1.columns if col in df2.columns])

        df_ls.append(pl.concat([df1, df2], how="diagonal_relaxed"))

    plot_powergain(pl.concat(df_ls), fname=fsearch.split("*")[1])


if __name__ == "__main__":
    run()
    run(fsearch="unifiedTI")
