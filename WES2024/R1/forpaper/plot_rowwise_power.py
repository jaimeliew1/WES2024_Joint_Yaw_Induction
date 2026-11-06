# dualitic is imported first to ensure correct monkey patching.
import dualitic

"""
R1 paper figures: turbine-level (row-wise) power of the 5 x 5 farm, LES against the wake models, <P>_row / <P>_row 1 against row.

Two figures, both with one panel per set of set points (the same wake models are fed the set points of the panel in a 5 x 1 forward
solve, see generate/generate_rowwise_power_models.py), so that differences between the curves are due to the wake model only:
    plot_rowwise_power_umm         2 x 3: No control, Thrust (TANDEM), Yaw (TANDEM), Joint (TANDEM / Vortex sheet / Gaussian); every wake model
                                   uses the same annulus BEM-LUT rotor ("UMM"), fed (pitch, TSR, yaw);
    plot_rowwise_power_tabulated   1 x 4: Yaw (TANDEM), Joint (TANDEM / Vortex sheet / Gaussian); the rotor is the tabulated two-turbine ABL
                                   LES rotor (Cp, Ct vs Ct', |yaw|), fed (Ct', yaw), i.e. the rotor-model error is removed.
Normalisation: every model curve is divided by the same wake model's No-control row-1 power with the same rotor; the LES by the
square_NoControl row-1 mean (No-control panel: row 1 of the same run, ratio per 10-minute block). Row 1 is the unwaked front row.
The LES is the solid black line with squares (no error bars: +-2 SE are <= 0.02, see generate/generate_les_farm_power.py).

Usage: python -m WES2024.R1.forpaper.plot_rowwise_power
"""

import matplotlib.pyplot as plt
import numpy as np

from WES2024.R1 import figuresettings as fs
from WES2024.R1.farm_control_cases import CASES
from WES2024.R1.figuresettings import CONTROL_LABELS, LES_COLOR, MODEL_LABELS, VARIANT_COLORS, VARIANT_LABELS, VARIANT_MARKERS
from WES2024.R1.generate import generate_rowwise_power_models as gen_models
from WES2024.R1.generate.generate_les_farm_power import (les_row_ratio_to_row0, les_row_stats, ratio_with_se)

VARIANTS = list(gen_models.VARIANTS)

PANELS_UMM = ["NoControl", "Thrust_TANDEM", "Yaw_TANDEM", "Joint_TANDEM", "Joint_Vortex", "Joint_Gauss"]
PANELS_TAB = ["Yaw_TANDEM", "Joint_TANDEM", "Joint_Vortex", "Joint_Gauss"]


def panel_title(key):
    c = CASES[key]
    if key == "NoControl":
        return CONTROL_LABELS[key]
    return f"{CONTROL_LABELS[c['control']]} ({MODEL_LABELS[c['model']]})"


def make_figure(rotor, panels, ncols, figsize, stem):
    models = gen_models.generate()  # cached Cp (5,) per rotor / case / wake model
    nc_m, nc_se = les_row_stats("NoControl")
    rows = np.arange(5)
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    rms = {}
    for j, (ax, key) in enumerate(zip(axes.flat, panels)):
        if key == "NoControl":
            les = les_row_ratio_to_row0(key)[0]
        else:
            m, se = les_row_stats(key)
            les = ratio_with_se(m, se, nc_m[0], nc_se[0])[0]
        ax.plot(rows + 1, les, "s-", color=LES_COLOR, ms=5, zorder=1, label="LES")
        rms[key] = {}
        for k, v in enumerate(VARIANTS):
            norm = models[f"{rotor}__{key}__{v}"] / models[f"{rotor}__NoControl__{v}"][0]
            rms[key][v] = float(np.sqrt(np.mean((norm - les) ** 2)))
            ax.plot(rows + 1, norm, "--", marker=VARIANT_MARKERS[v], color=VARIANT_COLORS[v], mfc="none", ms=5, zorder=3 + k, label=VARIANT_LABELS[v])
        ax.set_title(f"({chr(97 + j)}) {panel_title(key)}", fontsize=10, loc="left")
        ax.set_xticks(rows + 1)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Row")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\langle P\rangle / \langle P_1\rangle$")
        ax.set_ylim([0., 1.04])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # layout: legend at the very top, then the "Set points derived from:" note, then the panel titles (the note sits over the first panel
    # that has set points; No control has none)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / figsize[1]))
    plt.subplots_adjust(wspace=0.1)
    fig.canvas.draw()
    first = next(j for j, key in enumerate(panels) if key != "NoControl")
    ax0 = axes[0,0]
    top = ax0._left_title.get_window_extent().transformed(fig.transFigure.inverted()).y1
    fig.text(ax0.get_position().x0, top + 0.004, "Set points derived from:", fontsize=10, ha="left", va="bottom", color="0.3")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9)
    fs.save(stem)
    print(f"RMS(model - LES) over rows 1-5, {stem}")
    print(f"{'':16s}" + "".join(f"{VARIANT_LABELS[v]:>16s}" for v in VARIANTS))
    for key, r in rms.items():
        print(f"{key:16s}" + "".join(f"{r[v]:16.3f}" for v in VARIANTS))


def print_rotor_comparison():
    """Effect of the rotor model on the normalised row power: BEM-LUT annulus rotor (UMM) vs the LES-derived tabulated rotor (Cp, Ct(Ct', yaw)),
    for the same set points and wake model. Each rotor is normalised by its own No-control row 1 (as in the figures), so this is the
    change in the plotted curves, over rows 1-5 (RMS and max |difference|) and at row 1 (signed, tabulated - UMM)."""
    models = gen_models.generate()
    norm = lambda rotor, key, v: models[f"{rotor}__{key}__{v}"] / models[f"{rotor}__NoControl__{v}"][0]
    print("\nRotor model effect on normalised power: tabulated (LES look-up table) minus BEM-LUT (UMM), same set points & wake model")
    print(f"{'case':14s}{'yaw':>5s}" + "".join(f"{VARIANT_LABELS[v]:>34s}" for v in VARIANTS))
    print(f"{'':19s}" + "".join(f"{'RMS / max|d| / row-1 d':>34s}" for _ in VARIANTS))
    stats = {}
    for key in PANELS_UMM:
        yawed = key not in ("NoControl", "Thrust_TANDEM")
        cells = []
        for v in VARIANTS:
            d = norm("tabulated", key, v) - norm("umm", key, v)
            stats.setdefault(yawed, {}).setdefault(v, []).append(np.sqrt(np.mean(d**2)))
            cells.append(f"{np.sqrt(np.mean(d**2)):7.3f} /{np.abs(d).max():6.3f} /{d[0]:+7.3f}")
        print(f"{key:14s}{'yes' if yawed else 'no':>5s}" + "".join(f"{c:>34s}" for c in cells))
    print(f"{'mean RMS, no yaw misalignment (No control, Thrust):':56s}" + "".join(f"{np.mean(stats[False][v]):8.3f}" for v in VARIANTS))
    print(f"{'mean RMS, with yaw misalignment (Yaw, 3 x Joint):':56s}" + "".join(f"{np.mean(stats[True][v]):8.3f}" for v in VARIANTS))


def plot():
    make_figure("umm", PANELS_UMM, 3, (6.5, 4.25), "plot_rowwise_power_umm")
    make_figure("tabulated", PANELS_TAB, 4, (8, 2.7), "plot_rowwise_power_tabulated")
    print_rotor_comparison()


if __name__ == "__main__":
    plot()
