"""
Short script for visualizing turbine control set points and Cp

Kirby Heck
January 2025
"""

import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt

from WES2024.LES_new.step_3_optimize_controllers import LES_input_dir

LES_output_dir = Path(__file__).parent / "LES_output"

# LES_input_dir = Path(__file__).parent.parent / "LES" / "LES_newinput"

figpath = Path(__file__).parent / "calibration"  # / "old"
figpath.mkdir(exist_ok=True, parents=True)


def plot(df, name=None):
    """Plots set points C_T', yaw"""
    fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"], c=df["Cp"])
    ax.set_aspect(1)
    for x, y, ctp, yaw, Cp in df.select(["x", "y", "Ctprime", "yaw", "Cp"]).iter_rows():
        ax.text(
            x,
            y,
            f"$C_T' = {ctp:.1f}$,\n$\\gamma={yaw*180/np.pi:.0f}^\\circ$,\n$C_P={Cp:.2f}$",
            fontsize=6,
        )
    ax.set_title(name.split("_")[-1])
    plt.savefig(figpath / f"setpoints_{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_Cp(df, ax=None, name=None):
    """Plots model predicted Cp"""
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"], c=df["Cp"])
    ax.set_aspect(1)
    for x, y, Cp in df.select(["x", "y", "Cp"]).iter_rows():
        ax.text(x, y, f"{Cp:.2f}", ha="left", va="bottom", fontsize=10)

    if name is not None:
        plt.savefig(figpath / f"Cp_{name}.png", dpi=300, bbox_inches="tight")
        plt.close()


def run():
    for f in LES_input_dir.glob("*LESnew*.csv"):
        print(f.stem)
        df = pl.read_csv(f)
        plot(df, name=f.stem)


def plot_LES():
    for f in LES_output_dir.glob("*.csv"):
        print(f.name)
        df = pl.read_csv(f)
        plot_Cp(df, name=f.name)


def plot_for_jaime():
    rows = ["jointcontrol", "nocontrol", "thrustcontrol", "yawcontrol"]
    cols = ["linear", "niayifar"]

    fig, axarr = plt.subplots(
        figsize=(6, 12), nrows=len(rows), ncols=len(cols), sharex=True, sharey=True
    )
    for f in LES_input_dir.glob("*jaime*.csv"):
        parse = f.stem.split("_")
        ctrl = parse[3]
        superpos = parse[1]
        ax = axarr[rows.index(ctrl), cols.index(superpos)]
        print(f.name)
        df = pl.read_csv(f)

        plot_Cp(
            df,
            ax=ax,
        )

    for ax, row in zip(axarr[:, 0], rows):
        ax.set_ylabel(row)
    for ax, col in zip(axarr[0, :], cols):
        ax.set_title(col)

    plt.savefig(figpath / f"Cp_comparison_jaime.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run()
    # plot_for_jaime()
    # plot_LES()
