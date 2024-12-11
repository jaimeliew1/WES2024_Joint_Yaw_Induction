from WES2024.CustomRotors import CosineAD, UnifiedLUTAD
from UnifiedMomentumModel import Momentum
from mitwindfarm.Windfield import Uniform
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGPATH = Path(__file__).parent / "debug"

if __name__ == "__main__":
    c = CosineAD()
    u = UnifiedLUTAD()
    shapiro = Momentum.LimitedHeck()

    cosine = []
    cosine_ct = []
    unified = []
    yaws = np.arange(-50, 51, 10)
    # it appears this isn't vectorized so we will loop it
    for yaw in np.deg2rad(yaws):
        unified.append(u(0, 0, 0, Uniform(), 2.0, yaw).Cp)
        cosine.append(c(0, 0, 0, Uniform(), 2.0, yaw).Cp)
        cosine_ct.append(c(0, 0, 0, Uniform(), 2.0, yaw).Ct)

        if abs(yaw) < 1e-5:
            Cp0 = cosine[-1]
            Ct0 = cosine_ct[-1]

    fig, ax = plt.subplots()
    yaw_ax = np.linspace(-50, 50, 201)
    ax.scatter(yaws, cosine, label="Cosine")
    ax.scatter(yaws, unified, marker="x", label="Unified")
    ax.plot(yaw_ax, np.cos(np.deg2rad(yaw_ax)) ** 3 * Cp0, color="k", label="$\\cos^3(\\gamma)$")
    ax.legend()
    ax.set_xlabel("$\\gamma$ (deg.)")
    ax.set_ylabel("Power $C_P$")
    plt.savefig(FIGPATH / "test_cosine.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots()
    ax.scatter(yaws, cosine_ct, label="MIT Rotor: Shapiro $et~al.$ (2018)")
    ax.plot(
        yaw_ax,
        np.cos(np.deg2rad(yaw_ax)) ** 2 * Ct0,
        color="k",
        label="$\\propto \\cos^2(\\gamma)$",
    )
    ax.plot(
        yaw_ax,
        np.cos(np.deg2rad(yaw_ax)) ** 1 * Ct0,
        color="gray",
        ls="-.",
        label="$\\propto \\cos(\\gamma)$",
    )
    sol = shapiro(np.full_like(yaw_ax, 2.0), np.deg2rad(yaw_ax))
    ax.plot(yaw_ax, sol.Ct, color="tab:blue", ls="--", label="Momentum model")
    # This matches the C_T closure in Shapiro:
    ax.plot(
        yaw_ax,
        16
        * 2.0
        * np.cos(np.deg2rad(yaw_ax)) ** 2
        / (4 + 2.0 * np.cos(np.deg2rad(yaw_ax)) ** 2) ** 2,
        color="tab:purple",
        ls=":", 
        lw=2.5, 
        label="Shapiro closure $C_T(C_T', \\gamma)$"
    )

    ax.legend()
    ax.set_xlabel("$\\gamma$ (deg.)")
    ax.set_ylabel("Thrust $C_T$")

    plt.savefig(FIGPATH / "test_cosine_ct.png", dpi=200)
    plt.close()
