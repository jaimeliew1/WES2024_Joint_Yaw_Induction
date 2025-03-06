"""
Check turbine positions LES vs model

Kirby Heck
2024 December 18
"""

from pathlib import Path
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from WES2024.LES_new.compare_turbinelevel import plot_scatter_with_labels
from WES2024.LES_new.compare_superposition import (
    BASE_LAYOUT,
    read_LES_outputs,
    LES_output_dir,
)

figpath = Path(__file__).parent / "calibration"


def run():
    """Plot farm layouts"""
    df = read_LES_outputs(LES_output_dir.glob("*.csv")).filter(method="nocontrol")

    fig, ax = plt.subplots()
    plot_scatter_with_labels(df["x"] - df["x"].min(), df["y"] - df["y"].min(), df["turbine_id"], ax=ax, text="{:d}", marker="x")
    plot_scatter_with_labels(
        BASE_LAYOUT.x - np.min(BASE_LAYOUT.x),
        BASE_LAYOUT.y - np.min(BASE_LAYOUT.y),
        np.arange(25),
        ax=ax,
        text="{:d}",
        marker="+",
    )
    ax.plot([], ls='none', marker='x', label='LES', color='k')
    ax.plot([], ls='none', marker='+', label='MITWindfarm', color='k')
    ax.legend()
    plt.savefig(figpath / "farmlayout_comparison.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    run()
