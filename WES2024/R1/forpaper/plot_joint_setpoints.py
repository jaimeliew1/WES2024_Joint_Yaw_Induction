# dualitic is imported first to ensure correct monkey patching.
import dualitic

"""
R1 paper figure: the control set points of the farm-control study and the farm power coefficient the optimiser's own model predicts for them
(wdir 0, 5 x 5 aligned farm, 6 D spacing). One row per set of set points:
    TANDEM (with the ground) No control, Thrust, Yaw (tilde lambda; "YawMod"), Yaw K Omega^2 and Joint control,
    Vortex sheet Joint control, Gaussian Joint control.
Left: top-view schematic of the 5 x 1 column of set points that is tiled over the 5 columns (lambda, theta_p, gamma, C_T' of rows 1-4; the last
turbine, and every turbine for No control, is at its individually power-maximising set point, written with the symbols _opt). Right: model
C_P,farm of the 5 x 5 farm from the optimiser CSVs (data/generate_les_setpoints_tiled5x5/tiled), with the gain over the same model's No control:
one bar plot shared by the five TANDEM rows (bars level with their schematics), and a separate No-control / Joint bar plot for the Vortex sheet and
Gaussian models. The wake model is the title of each bar plot, the control strategy the label of each bar; the schematics carry no titles. Bar colour = wake model, hatch = control strategy (as plot_les_farm_gain.py). The realised LES gains are in
plot_les_farm_gain.py.

Usage: python -m WES2024.R1.forpaper.plot_joint_setpoints
"""

import matplotlib.pyplot as plt
import numpy as np

from WES2024.R1 import figuresettings as fs
from WES2024.R1.farm_control_cases import CASES, case_setpoints, model_nocontrol_setpoints
from WES2024.R1.figuresettings import CONTROL_HATCH, CONTROL_LABELS, MODEL_COLORS, MODEL_LABELS, NC_COLOR
from WES2024.R1.forpaper.setpoint_column import draw_setpoint_column

TANDEM_KEYS = ["NoControl", "Thrust_TANDEM", "YawMod_TANDEM", "Yaw_TANDEM", "Joint_TANDEM"]
OTHER_KEYS = ["Joint_Vortex", "Joint_Gauss"]
XMAX = 0.58
WSPACE = 0.1
FS_BAR = 10  # font size of the bar-plot text = that of the wake-model names
HSPACE = 0.4  # gridspec row gap in units of a row height
BAR_H = 0.38  # bar thickness in units of one schematic row, the same for every bar


def cp_farm(sp):
    return float(sp["Cp_farm"].iloc[0])


def bar_label(ax, v, y, cp_nc=None):
    gain = "" if cp_nc is None else rf" ({100 * (v / cp_nc - 1):+.0f}\%)"
    ax.text(v + 0.014, y, f"{v:.3f}{gain}", va="center", ha="left", fontsize=FS_BAR)


def style_bars(ax, ylim, last):
    ax.set_xlim(0, XMAX)
    ax.set_ylim(*ylim)
    ax.set_yticks([])
    ax.set_xticks([0, 0.2, 0.4])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    if last:
        ax.set_xlabel(r"$C_{P,\mathrm{farm}}$ (model)")
    else:
        ax.tick_params(axis="x", labelbottom=False)


def plot():
    keys = TANDEM_KEYS + OTHER_KEYS
    n = len(keys)
    fig = plt.figure(figsize=(8.6, 1.316 * n))
    gs = fig.add_gridspec(n, 2, width_ratios=[2.9, 1.0], wspace=WSPACE, hspace=HSPACE)

    for i, k in enumerate(keys):
        ax = fig.add_subplot(gs[i, 0])
        draw_setpoint_column(ax, case_setpoints(k), MODEL_COLORS[CASES[k]["model"]], all_opt=k == "NoControl")
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("$x/D$")

    # TANDEM: one bar plot level with the five schematics
    nt = len(TANDEM_KEYS)
    bx = fig.add_subplot(gs[:nt, 1])
    cp = [cp_farm(case_setpoints(k)) for k in TANDEM_KEYS]
    span = nt + (nt - 1) * HSPACE  # the shared axes spans nt rows and the gaps between them (row height = 1)
    ys = 1 - (np.arange(nt) * (1 + HSPACE) + 0.5) / span  # bar i level with the centre of schematic i, axes fraction
    for y, k, v in zip(ys, TANDEM_KEYS, cp):
        bx.barh(y, v, height=BAR_H / span, color=NC_COLOR if k == "NoControl" else MODEL_COLORS["TANDEM"], edgecolor="k",
                hatch=CONTROL_HATCH[CASES[k]["control"]])
        bar_label(bx, v, y, None if k == "NoControl" else cp[0])
    style_bars(bx, (0, 1), last=False)
    bx.set_yticks(ys)
    bx.set_yticklabels([CONTROL_LABELS[CASES[k]["control"]] for k in TANDEM_KEYS], fontsize=FS_BAR)
    bx.tick_params(axis="y", length=0)
    bx.set_title(MODEL_LABELS["TANDEM"], loc="left", fontsize=FS_BAR)
    print("TANDEM:", {k: round(v, 4) for k, v in zip(TANDEM_KEYS, cp)}, [f"{100 * (v / cp[0] - 1):+.1f}%" for v in cp])

    # Vortex sheet, Gaussian: No control and Joint
    for j, k in enumerate(OTHER_KEYS):
        m = CASES[k]["model"]
        bx = fig.add_subplot(gs[nt + j, 1])
        v_nc, v_j = cp_farm(model_nocontrol_setpoints(m)), cp_farm(case_setpoints(k))
        for y, v, color, hatch in ((0.72, v_nc, NC_COLOR, None), (0.28, v_j, MODEL_COLORS[m], CONTROL_HATCH["Joint"])):
            bx.barh(y, v, height=0.38, color=color, edgecolor="k", hatch=hatch)
        bar_label(bx, v_nc, 0.72)
        bar_label(bx, v_j, 0.28, v_nc)
        style_bars(bx, (0, 1), last=j == len(OTHER_KEYS) - 1)
        bx.set_yticks([0.72, 0.28])
        bx.set_yticklabels([CONTROL_LABELS["NoControl"], CONTROL_LABELS["Joint"]], fontsize=FS_BAR)
        bx.tick_params(axis="y", length=0)
        bx.set_title(MODEL_LABELS[m], loc="left", fontsize=FS_BAR)
        print(f"{MODEL_LABELS[m]}: No control {v_nc:.4f}  Joint {v_j:.4f}  predicted gain {100 * (v_j / v_nc - 1):+.1f}%")

    fs.save()


if __name__ == "__main__":
    plot()
