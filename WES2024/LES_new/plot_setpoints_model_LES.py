import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt

from WES2024.LES_new.step_3_optimize_controllers import LES_input_dir
from WES2024.LES_new.final_calibration import LES_output_dir, LES_pnormfact
from UnifiedMomentumModel.Momentum import UnifiedMomentum
from WES2024.LES_new.plot_row_power import controller_labels

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


def plot_setpoints_sidebyside(df, var="Cp", name="", clim=[0.5, 1.05], D=1.2, axs=None): 
    """Plot Cp side-by-side LES vs MITWindfarm"""
    df = df.sort(by=["simulator", "turbine"])
    # Get x-location for turbine 20 for each simulator and compute new x, y columns
    df_x0 = df.filter(pl.col("turbine") == 20).select(["simulator", pl.col("x").alias("x0"), pl.col("y").alias("y0")])
    df = df.join(df_x0, on="simulator", how="left").with_columns(
        (pl.col("x") - pl.col("x0")).alias("xnew"), 
        (pl.col("y") - pl.col("y0")).alias("ynew"), 
        (pl.col("yaw") * 180 / np.pi).alias("yawdeg"),
    )

    # make the figure(s)
    fig, axs = plt.subplots(figsize=(12, 4), ncols=3, sharex=True, sharey=True)

    ax = axs[0]
    sub = df.filter(simulator="LES")
    for x, y, ctp, yaw, yawdeg in sub.select(["xnew", "ynew", "Ctprime", "yaw", "yawdeg"]).iter_rows():
        # ax.text(x, y + 1, f"$({ctp:.2f}, {yawdeg:.0f}^\\circ)$", fontsize=4, ha='center', va='bottom')

        base_x = np.array([0, 0])
        base_y = np.array([-D/2, D/2])
        x_rot = base_x * np.cos(yaw) - base_y * np.sin(yaw)
        y_rot = base_x * np.sin(yaw) + base_y * np.cos(yaw)
        ax.plot(x_rot + x, y_rot + y, color='k', lw=1.5, alpha=0.3)

        ax.text(x + 0.7, y, f"{ctp:.2f}, \n${yawdeg:.0f}^\\circ$", fontsize=7, ha='left', va='center')

    controlname = controller_labels[name.split("_")[-1]]
    ax.set_title(f"{controlname} setpoints ($C_T', \\gamma$)")
    # ax.text(0.05, 0.95, controlname, transform=ax.transAxes, ha='left', va='top', fontsize=14)

    for ax, simulator in zip(axs[1:], ["MITWindfarm", "LES"]): 
        sub = df.filter(simulator=simulator)

        im = ax.scatter(sub["xnew"], sub["ynew"], c=sub[var], clim=clim)
        ax.set_title(simulator)
        xlim = np.array(ax.get_xlim())
        ylim = np.array(ax.get_ylim())
        ax.set_xlim(xlim + np.array([-2, 2]))
        ax.set_ylim(ylim + np.array([0, 1]))

        for x, y, val in sub.select(["xnew", "ynew", var]).iter_rows(): 
            ax.text(x, y+0.75, f"{val:.2f}", fontsize=10, ha='center', va='bottom')

    for ax in axs: 
        ax.set_aspect(1)
        ax.set_xlabel("$x/D$")
    axs[0].set_ylabel("$y/D$")
    plt.subplots_adjust(wspace=0.1)

    bbox = axs[2].get_position()
    cax = fig.add_axes([bbox.x0 + bbox.width + 0.02, bbox.y0, 0.01, bbox.height])
    plt.colorbar(im, label="$P/P_{\\mathrm{Betz}}$", shrink=0.6, cax=cax)
    plt.savefig(figpath / f"LES_model_turbineCp_{name}.png", dpi=300, bbox_inches='tight')
    plt.close()


def run(fsearch="*iter_01*.csv"):
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
        plot_setpoints_sidebyside(df, name=casename.split(".csv")[0], var="Pnorm")
        # break


if __name__ == "__main__": 
    run()
    # run(fsearch="*unifiedTI*.csv")

