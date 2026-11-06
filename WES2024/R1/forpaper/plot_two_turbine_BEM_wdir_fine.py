# dualitic is imported first to ensure correct monkey patching.
import dualitic

"""
R1 paper figure: two-turbine BEM set points against wind direction (6D aligned pair), all six control strategies.
The first panel shows the farm power change relative to the greedy (NoControl) farm in percent (Cp_farm is in the wake-model comparison figure).
Reads the outward-warm-started sweep of generate/generate_two_turbine_BEM_fine.py. The yaw strategies are split at wdir = 0 (two symmetric optima,
indeterminate there), so no line is drawn across the origin for them. The paper uses the vortex sheet.

Usage: python -m WES2024.R1.forpaper.plot_two_turbine_BEM_wdir_fine [--model varvortex|gaussBP|tandem]
Saves R1/figs/plot_two_turbine_BEM_wdir_<model>.png.
"""

import argparse

import matplotlib.pyplot as plt
import polars as pl

from WES2024.R1 import figuresettings as fs
from WES2024.R1.generate.generate_two_turbine_BEM_fine import load as load_fine

LEAD_TURBINE = 0  # upstream turbine (layout = Layout([0, 6], [0, 0]))
UNBROKEN = ("NoControl", "ThrustControl")  # no yaw variable: continuous through wdir = 0
METHODS = ["NoControl", "ThrustControl", "YawControl", "YawModControl", "YawKOmegaControl", "JointControl"]


def load(model="varvortex"):
    df = load_fine(model=model)
    farm = df.group_by("method", "wdir").agg(pl.col("Cp").mean().alias("Cp_farm"))
    # Farm power relative to the greedy (NoControl) farm at the same wdir, as a percent change. Cp is normalised by the
    # same freestream speed for every strategy, so the power ratio equals the Cp_farm ratio.
    greedy = farm.filter(pl.col("method") == "NoControl").select("wdir", pl.col("Cp_farm").alias("Cp_greedy"))
    farm = farm.join(greedy, on="wdir").with_columns(
        ((pl.col("Cp_farm") / pl.col("Cp_greedy") - 1) * 100).alias("dP_pct")
    )
    lead = (
        df.filter(pl.col("turbine") == LEAD_TURBINE)
        .with_columns(
            pl.col("yaw").degrees().alias("gamma_1"),
            pl.col("Ct").alias("Ct_1"),
            pl.col("Ctprime").alias("Ctprime_1"),
            pl.col("setpoint_0").degrees().alias("pitch_1"),
            pl.col("setpoint_1").alias("tsr_1"),
        )
        .select("method", "wdir", "gamma_1", "Ct_1", "Ctprime_1", "pitch_1", "tsr_1")
    )
    return farm.join(lead, on=("method", "wdir")).sort("method", "wdir")


def plot(df, name):
    # first-panel label uses \mathrm, not \text: matplotlib's usetex preamble has no amsmath
    panels = [
        ("dP_pct", r"$\Delta P_\mathrm{farm}$ (\%)"), 
        ("Ct_1", r"$C_{T}$"),
        ("gamma_1", r"$\gamma$ (deg)"), ("Ctprime_1", r"$C_{T}'$"),
        ("pitch_1", r"$\theta_{p}$ (deg)"), ("tsr_1", r"$\lambda$"),
    ]
    fig, axs = plt.subplots(3, 2, figsize=(5.5, 4.5), sharex=True)
    lines = []
    for i_panel, (ax, (col, label)) in enumerate(zip(axs.flat, panels)):
        ax.set_title(f"({chr(97 + i_panel)})", loc="left", fontsize=10, pad=3)  # (a)-(f), as in the wake-model figure
        for m in METHODS:
            d = df.filter(pl.col("method") == m).sort("wdir")
            segments = [d] if m in UNBROKEN else [d.filter(pl.col("wdir") < 0), d.filter(pl.col("wdir") > 0)]
            for i, seg in enumerate(segments):
                line, = ax.plot(seg["wdir"], seg[col], color=fs.CONTROLLER_COLORS[m], ls=fs.CONTROLLER_LINESTYLES[m],
                                label=fs.controller_labels[m] if i == 0 else None)
                if col == "dP_pct" and i == 0:  # the first segment carries the legend label
                    lines.append(line)
        ax.set_ylabel(label, fontsize=8 if col == "dP_pct" else None)  # that label is longer than the panel is tall

    for ax in axs[-1, :]:
        ax.set_xlabel("Wind direction (deg)")
    axs[0, 0].set_xlim(df["wdir"].min(), df["wdir"].max())

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.legend(handles=lines, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8)
    fs.save(name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="varvortex")
    args = ap.parse_args()
    plot(load(args.model), f"plot_two_turbine_BEM_wdir_{args.model}")
