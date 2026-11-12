"""
LES two-turbine power contours: farm (C_{P,farm}), leading (C_{P,1}) and waked (C_{P,2}) turbine power
coefficient as a function of leading-turbine thrust coefficient C_T' and yaw misalignment gamma_1.

Every panel shows the 110 LES set points. The farm panel (a) additionally marks

    - the global LES optimum (red star),
    - the set points statistically indistinguishable from it (sky-blue diamonds; two-sided Welch t-test on the
      10-minute window means, p > 0.05, see `R1/forpaper/twoturbine_gain_contours.indistinguishable_optima`), and
    - the C_T'(yaw) curves that four single-turbine control strategies trace out for one isolated yawed BEM rotor
      (`WES2024/R1/generate/generate_single_turbine_yaw_strategies.py`; computed for 0-45 deg of yaw).
      (^ currently commented out)

The legend is shared across the top of the figure.

Kirby Heck
2026
"""

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D

# from WES2024 import utils
from WES2024.R1 import figuresettings as fs
from WES2024.R1.generate import generate_single_turbine_yaw_strategies as strategies
from WES2024.R1.forpaper.twoturbine_gain_contours import (
    DATA_LES,
    LES_OPT_COLOR,
    LES_OPT_KW,
    _load,
    indistinguishable_optima,
    optimum,
    plot_contour,
)

# same order, colors and labels as R1/forpaper/plot_single_turbine_yaw_strategies.py
STRATEGIES = {
    "YawMod": ("YawModControl", r"Yaw (constant $\tilde{\lambda}$)"),
    "YawKOmega": ("YawKOmegaControl", r"Yaw ($K\Omega^2$)"),
    "Yaw": ("YawControl", r"Yaw (const. $\theta_{p,\mathrm{opt}}$, $\lambda_\mathrm{opt}$)"),
    "Joint": ("JointControl", "Joint"),
}

PANELS = [("Cp", r"$C_{P,\mathrm{farm}}$"), ("Cp_1", r"$C_{P,1}$"), ("Cp_2", r"$C_{P,2}$")]
# one color scale for all three panels so they can be compared directly
CONTOUR_KW = dict(vmin=0.3, vmax=0.6, extend="both", levels=np.arange(0.3, 0.601, 0.025), cmap="viridis")
STAR_KW = dict(s=60, marker="*", color="tab:red", lw=0.5, edgecolor="w", zorder=10)
LINE_HALO = [pe.Stroke(linewidth=2., foreground="w", alpha=0.8), pe.Normal()]


def make_plots():
    les = _load(DATA_LES).filter(pl.col("n_win").is_not_null())  # all 110 cases have power statistics
    stat = indistinguishable_optima(les)
    best_yaw, best_ctp = optimum(les)
    yaw_strat = strategies.generate()

    fig, axs = plt.subplots(ncols=3, figsize=(6.5, 2.5), sharex=True, sharey=True, layout="constrained")
    for k, (ax, (key, label)) in enumerate(zip(axs, PANELS)):
        ax.set_title(f"({chr(97 + k)}) {label}", fontsize=9, loc="left")
        im = plot_contour(les, ax, key=key, **CONTOUR_KW)
        ax.scatter(les["yaw"], les["ctp"], s=5, marker="o", color="w", edgecolor="k", lw=0.5, zorder=4)
        ax.set_xlabel(r"$\gamma_1$ (deg.)")

    ax = axs[0]
    ax.scatter(stat["yaw"], stat["ctp"], **LES_OPT_KW)
    ax.scatter(best_yaw, best_ctp, **STAR_KW)
    # for m, (ctrl, _) in STRATEGIES.items():
    #     d = yaw_strat.filter(pl.col("method") == m).sort("yaw")
    #     ax.plot(d["yaw"], d["Ctprime"], color=utils.controller_colors[ctrl], lw=1.2, path_effects=LINE_HALO, zorder=6)
    axs[0].set_ylabel("$C_{T,1}'$")
    axs[0].set_xlim(0, les["yaw"].max())

    # shared legend across the top: LES markers on one row, control strategies on the next
    marker_handles = [
        Line2D([], [], ls="", marker="o", markersize=2.5, markerfacecolor="w", markeredgecolor="k", markeredgewidth=0.5),
        Line2D([], [], ls="", marker="*", markersize=8, markerfacecolor="tab:red", markeredgecolor="w", markeredgewidth=0.5),
        Line2D([], [], ls="", marker="D", markersize=4.5, color=LES_OPT_COLOR, alpha=0.8),
    ]
    marker_labels = ["LES set points", "Global LES optimum", "Statistical LES optima"]
    # line_handles = [
    #     Line2D([], [], color=utils.controller_colors[ctrl], lw=1.2)
    #     for ctrl, _ in STRATEGIES.values()
    # ]
    # line_labels = [lab for _, lab in STRATEGIES.values()]

    # constrained layout sizes the margins to the content; "outside" legends get their own reserved rows
    fig.colorbar(im, ax=axs, shrink=0.9, pad=0.02, ticks=np.arange(0.3, 0.61, 0.1), label=r"Power coefficient $C_{P,i}$")
    # one legend, two rows: matplotlib fills columns first, so interleave the rows and pad the short top row
    blank = Line2D([], [], alpha=0)
    # handles = [h for pair in zip(marker_handles + [blank], line_handles) for h in pair]
    # labels = [l for pair in zip(marker_labels + [""], line_labels) for l in pair]
    handles = [h for pair in zip(marker_handles + [blank]) for h in pair]
    labels = [l for pair in zip(marker_labels + [""]) for l in pair]
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False, fontsize=8,
               columnspacing=1.2, handlelength=1.5)

    fs.save(pad_inches=0.02, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    make_plots()
