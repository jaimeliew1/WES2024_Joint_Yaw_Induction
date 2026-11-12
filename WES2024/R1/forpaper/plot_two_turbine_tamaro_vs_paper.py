"""
R1: our two-turbine test in the setup of Tamaro et al. (WES 2024), side by side with their results (their Fig. 22),
to show the qualitative agreement. Same four quantities in both rows, as functions of wind direction:
(a, e) leading-turbine TSR, (b, f) leading-turbine blade pitch, (c, g) leading-turbine |yaw|, (d, h) change in the
combined power of the two turbines relative to greedy control (NoControl), in percent.

Top row(s): this study (BEM rotor, see generate_two_turbine_tamaro.py), one row per wake model of the figure (STACKS: the
original axisymmetric Gaussian above the BP2016 Gaussian; the other models get a figure each). Model rows share y limits.
Bottom row: Tamaro et al., digitized from their Fig. 22 (data/extracted/tamaro_fig22/, not committed; without it only the model rows are plotted): their "cosine" and "optimal"
model curves. These are drawn with the colors/line styles of our nearest strategy (cosine <-> yaw with K Omega^2 torque
law; optimal <-> joint control); that pairing is our reading of their two cases. The y axes of the two rows deliberately
differ (different turbine, different wake model); only the shapes are to be compared. Their pitch is only available
for the optimal case. The "Greedy" line (our No Control) is constructed: TSR and pitch at their value at the largest
|wind direction|, zero yaw and zero power change. Digitization leaves small baseline offsets (e.g. ~1 deg of yaw and ~0.2 % of power far from the
wake). Their LES points (Fig. 22f) are not shown (SHOW_LES).

Usage: python -m WES2024.R1.forpaper.plot_two_turbine_tamaro_vs_paper
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from WES2024 import utils
from WES2024.R1 import figuresettings as fs
from WES2024.R1.generate.generate_two_turbine_tamaro import SPACING, TIAMB, methods
from WES2024.R1.forpaper.plot_two_turbine_tamaro import load
from WES2024.R1.wakemodels import DISPLAY_NAMES

FIGSIZE = (7.0, 4.25)  # inches, for 2 rows; paper text width is ~6.5-7 in
FS = 8  # label / tick / title font size
DIGITIZED = utils.CACHEDIR / "extracted" / "tamaro_fig22"
STACKS = [["gauss", "gaussBP"]]  # model rows of each figure
ROW_NAMES = {"gauss": "Initial submission: Axisymmetric Gaussian", "gaussBP": "Current revision: Gaussian (Bastankhah and Port\\'e-Agel, 2016)"}
SHOW_LES = False  # their LES points of Fig. 22f
XLIM = (-20, 20)

# (our column, their file, their column, axis label); P-panel columns come from the change relative to NoControl
PANELS = [
    ("tsr", "fig22a_combined", "tsr", r"TSR $\lambda$"),
    ("pitch", "fig22b_optimal", "pitch_deg", r"Blade pitch (deg)"),
    ("absyaw", "fig22c_combined", "yaw_deg", r"$|\gamma|$ (deg)"),
    ("dtotal", "fig22f_combined", "P_diff", r"$\Delta P$ total (\%)"),
]
# their model curves -> the strategy whose style they borrow, and their legend label
PAPER_CASES = {"model_cosine": ("YawKOmegaControl", "Cosine"), "model_optimal": ("JointControl", "Optimal")}


def paper_available():
    """True if all digitized curves exist; warns and returns False otherwise, so the figure is made from the generatable data only."""
    missing = [fname for _, fname, _, _ in PANELS if not (DIGITIZED / f"{fname}.csv").exists()]
    if missing:
        warnings.warn(f"Digitized Tamaro et al. curves not found in {DIGITIZED}: {missing}. Plotting this study's data only.")
    return not missing


def paper(fname):
    df = pl.read_csv(DIGITIZED / f"{fname}.csv")
    if "source" not in df.columns:  # the pitch file only has the optimal case
        df = df.with_columns(source=pl.lit("model_optimal"))
    return df


def greedy_level(fname, d, their):
    """Greedy (unyawed, optimal TSR and pitch) reference for one of their panels: zero yaw and zero power change; TSR and
    pitch from their optimal-case value at the largest |wind direction|, where the wake no longer matters."""
    if fname in ("fig22c_combined", "fig22f_combined"):
        return 0.0
    c = d.filter(pl.col("source") == "model_optimal")
    return c[their][c["wd"].abs().arg_max()]


def plot(models):
    loaded = [load(m) for m in models]
    if any(x is None for x in loaded):
        return
    show_paper = paper_available()
    nrow = len(models) + show_paper

    fig = plt.figure(figsize=(FIGSIZE[0], FIGSIZE[1] * nrow / 2), constrained_layout=True)
    subfigs = fig.subfigures(nrow, 1, squeeze=False).ravel()
    for sf, model in zip(subfigs, models):
        sf.suptitle(f"{ROW_NAMES.get(model, DISPLAY_NAMES[model])}, {SPACING:g} D spacing, TI = {TIAMB * 100:g} \\%",
                    fontsize=FS + 1)
    if show_paper:
        subfigs[-1].suptitle("Tamaro et al. (2024), Fig. 22 (digitized)", fontsize=FS + 1)
    axs_rows = [sf.subplots(1, 4, sharex=True) for sf in subfigs]
    axs_models = axs_rows[: len(models)]

    for axs_top, (df, P) in zip(axs_rows, loaded):
        for col, (ax, (our, _, _, label)) in enumerate(zip(axs_top, PANELS)):
            for method in methods:
                style = dict(color=fs.CONTROLLER_COLORS[method], ls=fs.CONTROLLER_LINESTYLES[method], label=fs.controller_labels[method])
                if our == "dtotal":
                    if method == "NoControl":
                        continue
                    d = P.filter(pl.col("method") == method).sort("wdir")
                else:
                    d = df.filter((pl.col("method") == method) & (pl.col("turbine") == 0)).sort("wdir")  # leading turbine
                ax.plot(d["wdir"], d[our], **style)
    if len(models) > 1:  # common y limits for the model rows, column by column
        for col in range(len(PANELS)):
            lims = [axs[col].get_ylim() for axs in axs_models]
            for axs in axs_models:
                axs[col].set_ylim(min(l[0] for l in lims), max(l[1] for l in lims))

    for ax, (_, fname, their, label) in zip(axs_rows[-1], PANELS) if show_paper else ():
        d = paper(fname)
        ax.plot(XLIM, [greedy_level(fname, d, their)] * 2, color=fs.CONTROLLER_COLORS["NoControl"],
                ls=fs.CONTROLLER_LINESTYLES["NoControl"], label="Greedy")  # first, so the legend order matches the top row
        for source, (strategy, name) in PAPER_CASES.items():
            c = d.filter(pl.col("source") == source).sort("wd")
            if not c.is_empty():
                ax.plot(c["wd"], c[their], color=fs.CONTROLLER_COLORS[strategy], ls=fs.CONTROLLER_LINESTYLES[strategy], label=name)
        if SHOW_LES and fname == "fig22f_combined":
            for source, (strategy, name) in PAPER_CASES.items():
                c = d.filter(pl.col("source") == source.replace("model", "LES"))
                ax.plot(c["wd"], c[their], "o", ms=2.5, color=fs.CONTROLLER_COLORS[strategy], label=f"LES {name.lower()}")

    for row, axs in enumerate(axs_rows):
        for col, (ax, (_, _, _, label)) in enumerate(zip(axs, PANELS)):
            ax.set_title(f"({chr(97 + 4 * row + col)})", loc="left", fontsize=FS + 1)
            ax.set_ylabel(label, fontsize=FS)
            ax.set_xlabel("Wind direction (deg)", fontsize=FS)
            ax.tick_params(labelsize=FS - 1)
            ax.set_xlim(*XLIM)
            ax.grid(alpha=0.3)
            if col == 3:
                ax.axhline(0, color="k", lw=0.5)
        handles, labels = axs[0].get_legend_handles_labels()
        subfigs[row].legend(handles, labels, loc="outside lower center", ncol=3, fontsize=FS - 1, frameon=False)

    # the model-only figure gets its own name so it never replaces the full one
    fs.save(f"{Path(__file__).stem.replace('_vs_paper', '')}_{'_'.join(models)}{'_vs_paper' if show_paper else ''}", dpi=200)


if __name__ == "__main__":
    for models in STACKS:
        plot(models)
