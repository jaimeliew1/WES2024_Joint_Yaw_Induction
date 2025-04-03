import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt

from WES2024.LES_new.step_3_optimize_controllers import LES_input_dir
from WES2024.LES_new.final_calibration import LES_output_dir, LES_pnormfact
from UnifiedMomentumModel.Momentum import UnifiedMomentum

Betz = UnifiedMomentum()(2.0, 0).Cp

figpath = Path(__file__).parent / "figs"
figpath.mkdir(exist_ok=True, parents=True)


def plot_scatters(df, vars=["yaw", "Ctprime", "Cp"], name=""): 
    """Plot scatters of LES vs MITWindfarm"""
    df = df.sort(by=["simulator", "turbine"])
    fig, axs = plt.subplots(figsize=(2*len(vars), 2), ncols=len(vars))

    for ax, var in zip(axs, vars): 
        ax.scatter(df.filter(simulator="MITWindfarm")[var], df.filter(simulator="LES")[var])
        ax.set_title(var)
        ax.set_xlabel("MITWindfarm")
        ylim = ax.get_ylim()
        ax.plot(ylim, ylim, color='k', ls='--')
        ax.set_aspect(1.)

    axs[0].set_ylabel("LES")
    plt.tight_layout()
    plt.savefig(figpath / f"LES_model_comparison_{name}.png", dpi=300)
    plt.close()


def plot_Cp_sidebyside(df, var="Cp", name="", clim=[0.5, 1.05]): 
    """Plot Cp side-by-side LES vs MITWindfarm"""
    df = df.sort(by=["simulator", "turbine"])
    fig, axs = plt.subplots(figsize=(6, 2.5), ncols=2)

    for ax, simulator in zip(axs, ["LES", "MITWindfarm"]): 
        sub = df.filter(simulator=simulator)
        im = ax.scatter(sub["x"], sub["y"], c=sub[var], clim=clim)
        ax.set_title(simulator)
        ax.set_xlabel("$x/D$")
        ax.set_aspect(1)
        xlim = np.array(ax.get_xlim())
        ax.set_xlim(xlim + np.array([-5, 5]))

        for x, y, Cp in sub.select(["x", "y", var]).iter_rows(): 
            ax.text(x, y, f"{Cp:.2f}")

    axs[0].set_ylabel("$y/D$")
    plt.tight_layout()
    plt.colorbar(im, label="$P/P_{\\mathrm{Betz}}$", ax=axs)
    plt.savefig(figpath / f"LES_model_turbineCp_{name}.png", dpi=300)
    plt.close()


def run(fsearch="*LESnew*.csv"):
    for f in LES_output_dir.glob(fsearch):
        casename = f.name.split('LES_')[-1]
        print("Running:", casename)
        try: 
            model_inputs = next(LES_input_dir.glob(f"*{casename}"))
        except StopIteration as e: 
            print("\tNo model inputs found")
            continue

        df1 = pl.read_csv(f)
        df1 = df1.with_columns(
            pl.lit("LES").alias("simulator"), 
            pl.col("turbine_id").alias("turbine"), 
            (pl.col("Cp") / LES_pnormfact).alias("Pnorm"), 
        ).drop("turbine_id")
        df2 = pl.read_csv(model_inputs).with_columns(
            (pl.col("Cp") / Betz).alias("Pnorm"), 
        )
        df = pl.concat([df1, df2], how="diagonal_relaxed")
        plot_Cp_sidebyside(df, name=casename.split(".csv")[0], var="Pnorm")

if __name__ == "__main__": 
    run()
    # run(fsearch="*unifiedTI*.csv")

