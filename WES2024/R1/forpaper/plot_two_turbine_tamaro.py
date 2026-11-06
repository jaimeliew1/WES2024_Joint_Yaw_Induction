"""
R1: two-turbine test in the setup of Tamaro et al. (WES 2024); see generate_two_turbine_tamaro.py. One figure per
wake model (2 x 3): TSR, blade pitch and |yaw| of the leading turbine; change in leading, downstream and total power
relative to greedy control (NoControl), in percent.

Usage: python -m WES2024.R1.forpaper.plot_two_turbine_tamaro
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from WES2024 import utils
from WES2024.R1 import figuresettings as fs
from WES2024.R1.generate.generate_two_turbine_tamaro import FILESTEM, MODELS, SPACING, TIAMB, methods
from WES2024.R1.wakemodels import DISPLAY_NAMES


def load(model):
    """(df, P) for one wake model: df = per-turbine set points (tsr, pitch [deg], absyaw [deg], Cp, ...) per method and
    wdir; P = lead / downstream / total Cp and their change relative to NoControl in percent (dlead, ddown, dtotal).
    None if the cache is missing."""
    fn = utils.CACHEDIR / f"{FILESTEM}_{model}.csv"
    if not fn.exists():
        print(f"Missing cache for {model}; run generate_two_turbine_tamaro.py")
        return None
    df = pl.read_csv(fn).with_columns(
        pitch=np.rad2deg(pl.col("setpoint_0")), tsr=pl.col("setpoint_1"), absyaw=np.rad2deg(pl.col("yaw")).abs()
    )
    P = df.pivot(columns="turbine", index=["method", "wdir"], values="Cp").rename({"0": "lead", "1": "down"})
    P = P.with_columns(total=pl.col("lead") + pl.col("down"))
    greedy = P.filter(pl.col("method") == "NoControl").select("wdir", pl.all().exclude("method", "wdir").name.suffix("_g"))
    P = P.join(greedy, on="wdir").with_columns(
        [(100 * (pl.col(k) - pl.col(f"{k}_g")) / pl.col(f"{k}_g")).alias(f"d{k}") for k in ("lead", "down", "total")]
    )
    return df, P


def plot(model):
    loaded = load(model)
    if loaded is None:
        return
    df, P = loaded

    fig, axs = plt.subplots(2, 3, figsize=(9, 4.5), sharex=True, constrained_layout=True)
    top = [("tsr", r"TSR $\lambda$"), ("pitch", "Blade pitch (deg)"), ("absyaw", r"$|\gamma|$ (deg)")]
    bottom = [("dlead", r"$\Delta P$ leading (\%)"), ("ddown", r"$\Delta P$ downstream (\%)"), ("dtotal", r"$\Delta P$ total (\%)")]
    for method in methods:
        style = dict(color=fs.CONTROLLER_COLORS[method], ls=fs.CONTROLLER_LINESTYLES[method], label=fs.controller_labels[method])
        d = df.filter((pl.col("method") == method) & (pl.col("turbine") == 0)).sort("wdir")  # leading turbine only
        for ax, (col, _) in zip(axs[0], top):
            ax.plot(d["wdir"], d[col], **style)
        if method != "NoControl":
            d = P.filter(pl.col("method") == method).sort("wdir")
            for ax, (col, _) in zip(axs[1], bottom):
                ax.plot(d["wdir"], d[col], **style)
    for ax, (_, label) in zip(np.r_[axs[0], axs[1]], top + bottom):
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    for ax in axs[1]:
        ax.set_xlabel("Wind direction (deg)")
        ax.axhline(0, color="k", lw=0.5)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"{DISPLAY_NAMES[model]}, {SPACING:g} D spacing, TI = {TIAMB * 100:g} \\%")
    fs.save(f"{Path(__file__).stem}_{model}", dpi=200)


if __name__ == "__main__":
    for model in MODELS:
        plot(model)
