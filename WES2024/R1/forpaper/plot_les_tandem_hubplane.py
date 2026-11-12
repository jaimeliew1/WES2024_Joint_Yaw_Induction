# dualitic is imported first to ensure correct monkey patching.
import dualitic

"""
R1 paper figure: hub-height velocity of the 5x5 farm under joint control, the LES (a) against TANDEM (b).
Same figure as the MITWindfarm-vs-LES schematic of the original paper, with the TANDEM wake model in place of the Gaussian wake
model of MITWindfarm, and for the joint-control set points optimised with TANDEM.

Both panels are u / u_inf at z = z_hub, u_inf the hub-height speed of the LES precursor: time-mean u for the LES and the
TANDEM forward solve (with ground, coupled vortex decay, LES inflow profiles, rotors at the LES's own Ct' and yaw) interpolated to the LES
grid. The rotors are drawn as lines of length D along the rotor plane, rotated by their yaw.

Data: LES_DATA/R1/les_hubplane_Joint_TANDEM.npz and data/generate_les_tandem_hubplane.npz (generate/generate_les_tandem_hubplane.py).

Usage: python -m WES2024.R1.forpaper.plot_les_tandem_hubplane
"""

import matplotlib.pyplot as plt
import numpy as np

from WES2024.R1 import figuresettings as fs
from WES2024.R1.generate import generate_les_tandem_hubplane as gen

CMAP = "RdYlBu_r"
CLIM = (0.35, 1.65)  # as the original figure: centred on the free-stream value, u / u_inf = 1
XLIM = (-5, 39)
YLIM = (-4, 30)


def draw_rotors(ax, turbines, **kw):
    """Rotor discs seen from above: a line of length 1 D centred on (x, y), rotated by yaw (positive counter-clockwise)."""
    x, y, yaw = turbines[:, 0], turbines[:, 1], turbines[:, 3]
    ax.plot([x - 0.5 * np.sin(yaw), x + 0.5 * np.sin(yaw)], [y + 0.5 * np.cos(yaw), y - 0.5 * np.cos(yaw)],
            color="k", lw=0.9, solid_capstyle="butt", **kw)


def plot(d):
    x, y = d["x"], d["y"]
    dx, dy = x[1] - x[0], y[1] - y[0]
    extent = [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2]

    fig, axs = plt.subplots(figsize=(7, 2.5), ncols=2, sharey=True, sharex=True)
    k = 97
    for ax, key, title in zip(axs, ("u_les", "u_tandem"), ("LES", "TANDEM wake model")):
        im = ax.imshow(d[key].T, origin="lower", extent=extent, cmap=CMAP, clim=CLIM, interpolation="nearest")
        draw_rotors(ax, d["turbines"])
        ax.text(0.02, 0.98, f"({chr(k)}) {title}", fontsize=10, va="top", ha="left", transform=ax.transAxes)
        ax.set_xlabel("$x/D$")
        k += 1

    axs[0].set_xlim(XLIM)
    axs[0].set_ylim(YLIM)
    axs[0].set_ylabel("$y/D$")

    plt.subplots_adjust(right=0.8, wspace=0.1)
    bbox = axs[1].get_position()
    cax = fig.add_axes([bbox.x1 + 0.02, bbox.y0, 0.01, bbox.y1 - bbox.y0])
    plt.colorbar(im, cax=cax, label=r"$\bar{u}/u_\infty$")
    fs.save()


if __name__ == "__main__":
    plot(gen.generate())
