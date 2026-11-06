# dualitic is imported first to ensure correct monkey patching.
import dualitic

"""
R1 paper figure: realised gain in farm power of the 5 x 5 farm LES for every set of control set points, wdir 0.

Bars: Cp_farm(LES, control) / Cp_farm(LES, square_NoControl) - 1, +-2 SE (10-minute blocks, ratio of two independent runs, first-order
propagation; generate/generate_les_farm_power.py). Bar color = wake model that optimised the set points, hatch = control strategy
(Thrust: dots, Yaw: circles, Joint: slashes). Error-bar color = Welch-test group: bars whose error bars share a color are not
significantly different (Welch's t-test on the farm C_P of the 10-minute blocks, p > ALPHA, as print_les_farm_welch.py); black = a
run that differs from every other run. LES only: the gains the models predict for the same set points are in the set-point figures
(plot_joint_setpoints.py).

Usage: python -m WES2024.R1.forpaper.plot_les_farm_gain
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

from WES2024.R1 import figuresettings as fs
from WES2024.R1.figuresettings import CONTROL_HATCH, CONTROL_LABELS, MODEL_COLORS, MODEL_LABELS
from WES2024.R1.farm_control_cases import CASES
from WES2024.R1.generate.generate_les_farm_power import generate, les_farm_stats, ratio_with_se

KEYS = ["Thrust_TANDEM", "YawMod_TANDEM", "Yaw_TANDEM", "Joint_TANDEM", "Joint_Vortex", "Joint_Gauss"]
ALPHA = 0.05
GROUP_COLORS = ["#D62728", "#7B2D8E", "#8C564B"]  # error bars of the groups with >1 member, in bar order
SINGLE_COLOR = "k"  # error bars of runs that differ from all others


def welch_groups(keys):
    """Group index per key: connected components of the 'not significantly different' (p > ALPHA) graph of the block farm C_P."""
    blocks = {k: generate(k)["blocks"].mean(axis=0) for k in keys}
    same = {(a, b): stats.ttest_ind(blocks[a], blocks[b], equal_var=False).pvalue > ALPHA for a, b in itertools.combinations(keys, 2)}
    group = {k: i for i, k in enumerate(keys)}
    for (a, b), s in same.items():
        if s:
            ga, gb = group[a], group[b]
            group = {k: ga if g == gb else g for k, g in group.items()}
    for (a, b), s in same.items():  # the grouping is only a clean story if 'not different' is transitive
        if not s and group[a] == group[b]:
            print(f"WARNING: {a} and {b} are in one group but differ significantly")
    return [group[k] for k in keys]


def plot():
    nc, nc_se = les_farm_stats("NoControl")
    gain, err = [], []
    for k in KEYS:
        cp, se = les_farm_stats(k)
        r, r_se = ratio_with_se(cp, se, nc, nc_se)
        gain.append(100 * (r - 1))
        err.append(200 * r_se)
    gain, err = np.array(gain), np.array(err)
    groups = welch_groups(KEYS)
    shared = [g for g in dict.fromkeys(groups) if groups.count(g) > 1]
    ecolor = [GROUP_COLORS[shared.index(g)] if g in shared else SINGLE_COLOR for g in groups]

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = np.arange(len(KEYS))
    for xi, k, g, e, ec in zip(x, KEYS, gain, err, ecolor):
        c = CASES[k]
        ax.bar(xi, g, width=0.62, color=MODEL_COLORS[c["model"]], edgecolor="k", hatch=CONTROL_HATCH[c["control"]],
               zorder=2)
        for c_, lw in (("w", 2.8), (ec, 1.6)):  # white halo so the color reads on any bar
            ax.errorbar(xi, g, yerr=e, fmt="none", ecolor=c_, elinewidth=lw, capsize=4.5, capthick=lw, zorder=3)
        if g >= 0:
            ax.text(xi, g + e + 0.5, f"{g:+.1f}\\%", ha="center", va="bottom", fontsize=8)
        else:  # below the error bar, away from the model marker
            ax.text(xi, g - e - 0.5, f"${g:+.1f}\\%$", ha="center", va="top", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{CONTROL_LABELS[c['control']]}\n({MODEL_LABELS[c['model']]})" for c in (CASES[k] for k in KEYS)], fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlabel("Control strategy (Model used in optimizer)")
    ax.set_ylabel(r"Change in LES $C_{P,\mathrm{farm}}$ (\%)")
    ax.set_ylim(min(0, np.min(gain - err)) - 3.0, 1.15 * np.max(gain + err))
    handles = [Line2D([], [], color=GROUP_COLORS[i], lw=1.6) for i in range(len(shared))] + [Line2D([], [], color=SINGLE_COLOR, lw=1.6)]
    labels = [", ".join(CONTROL_LABELS[CASES[k]["control"]] + f" ({MODEL_LABELS[CASES[k]['model']]})"
                        for k, gk in zip(KEYS, groups) if gk == g) for g in shared] + ["Differs from all"]
    ax.legend(handles, labels, title=f"Error bars: same color = not significantly different (Welch, $p > {ALPHA}$)",
              fontsize=6.5, title_fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=len(labels), frameon=False,
              handlelength=1.2, columnspacing=1.0)
    fs.save()
    for k, g, e in zip(KEYS, gain, err):
        print(f"{k:14s} LES gain {g:6.2f}% +-{e:.2f} (2SE)  Welch group {groups[KEYS.index(k)]}")


if __name__ == "__main__":
    plot()
