"""
R1 paper figure: power-yaw (a) and thrust-yaw (b) relation of the leading turbine in the two-turbine LES (sheared, veered
ABL inflow) for Ct' = 0.8, 2 and 4, against the Unified Momentum Model (UMM) and the uniform-inflow LES the UMM was validated
against. Every curve is normalized by its own value at gamma = 0, and the cosine law X(gamma) / X(0) = cos^p(gamma) is fitted
to the ABL LES for each Ct' (exponents are in the legends and printed). Absolute values at gamma = 0 are printed as well.

Quantities, normalized with the case's rotor-equivalent wind speed REWS (column `rews`):
    C_P  = P / (1/2 rho A REWS^3)                  (LES_DATA/twoturbine_alldata_LES_std.csv)
    C_T  = T / (1/2 rho A REWS^2) = Ct' (ud / REWS)^2
The thrust is not stored, but the filtered actuator disk applies T = 1/2 A Ct' ud^2 (normal to the disk), with `ud` the disk
velocity. This reproduces the volume integral of the xAD budget term to within 0.7% (checked for
yaw = 0, 25 and 45 deg at Ct' = 2) and the cached power exactly; no filter-correction factor M enters.

Usage: python -m WES2024.R1.forpaper.plot_power_yaw_twoturbine
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from UnifiedMomentumModel.Momentum import UnifiedMomentum

from WES2024.R1 import figuresettings as fs
from WES2024.R1.les_data import UNIFORM_INFLOW_LES, load_twoturbine_lead

CTPRIMES = (0.8, 2.0, 4.0)  # local thrust coefficients of the leading turbine
YAW_MAX = 45  # deg, the largest yaw of the ABL LES

KW_UNI = dict(marker="o", facecolor="none", s=20, clip_on=False)
KW_ABL = dict(marker="^", facecolor="none", s=22, clip_on=False)


def cosine(gamma_deg, p):
    return np.cos(np.deg2rad(gamma_deg)) ** p


def fit_cosine(yaw, y):
    """Exponent p of the least-squares fit y = cos^p(yaw)."""
    (p,), _ = curve_fit(cosine, yaw, y, p0=2.0)
    return p


def plot():
    uni_all = pl.read_csv(UNIFORM_INFLOW_LES)
    umm = UnifiedMomentum()
    g = np.linspace(0, YAW_MAX, 91)
    cids = (np.array(CTPRIMES) - min(CTPRIMES)) / (max(CTPRIMES) - min(CTPRIMES))
    colors = sns.color_palette("flare", as_cmap=True)(cids)

    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.75), constrained_layout=True)
    for ctp, color in zip(CTPRIMES, colors):
        abl = load_twoturbine_lead(ctp).rename({"Cp_1": "Cp"})
        uni = uni_all.filter(((pl.col("ctp") - ctp).abs() < 1e-6) & (pl.col("yaw") <= YAW_MAX)).sort("yaw")
        uni = uni.rename({"cp": "Cp", "ct": "Ct"})
        sol = umm(ctp * np.ones_like(g), np.deg2rad(g))
        model = {"Cp": np.asarray(sol.Cp), "Ct": np.asarray(sol.Ct)}

        p = {}
        for ax, q in zip(axs, ("Cp", "Ct")):
            y_abl, y_uni = abl[q].to_numpy(), uni[q].to_numpy()
            p[q] = fit_cosine(abl["yaw"].to_numpy().astype(float), y_abl / y_abl[0])
            ax.plot(g, model[q] / model[q][0], color=color, lw=1.2)
            ax.plot(g, cosine(g, p[q]), color=color, ls="--", lw=1, label=rf"$C_T'={ctp:.1f}$: $\cos^{{{p[q]:.2f}}}\gamma$")
            ax.scatter(uni["yaw"], y_uni / y_uni[0], edgecolor=color, **KW_UNI)
            ax.scatter(abl["yaw"], y_abl / y_abl[0], edgecolor=color, **KW_ABL)
        print(
            f"Ct' = {ctp:.1f}: cosine exponents (ABL LES)  Pp = {p['Cp']:.2f}, Pt = {p['Ct']:.2f}\n"
            f"    at gamma = 0   Cp: UMM {model['Cp'][0]:.3f}, uniform LES {uni['Cp'][0]:.3f}, ABL LES {abl['Cp'][0]:.3f}"
            f"   Ct: UMM {model['Ct'][0]:.3f}, uniform LES {uni['Ct'][0]:.3f}, ABL LES {abl['Ct'][0]:.3f}"
        )

    for ax, lab, q in zip(axs, "ab", "PT"):
        ax.set_xlabel(r"Yaw misalignment $\gamma$ ($^\circ$)")
        ax.set_ylabel(rf"$C_{q}(\gamma)/C_{q}(0)$")
        ax.set_xlim(0, YAW_MAX)
        ax.set_xticks(np.arange(0, YAW_MAX + 1, 15))
        ax.text(-0.15, 1.0, f"({lab})", fontsize=9, ha="right", va="center", transform=ax.transAxes)
        ax.legend(fontsize=8, loc="lower left", handlelength=1.6)

    shared = [
        Line2D([], [], color="k", ls="", marker="o", mfc="none", ms=4.5, label="Uniform inflow LES"),
        Line2D([], [], color="k", ls="", marker="^", mfc="none", ms=4.5, label="ABL LES"),
        Line2D([], [], color="k", ls="-", label="UMM"),
        Line2D([], [], color="k", ls="--", lw=1, label="Cosine fit to ABL LES"),
    ]
    fig.legend(handles=shared, loc="outside upper center", ncol=4, fontsize=8, frameon=False, columnspacing=1.2)
    fs.save()


if __name__ == "__main__":
    plot()
