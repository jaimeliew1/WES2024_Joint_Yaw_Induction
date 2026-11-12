"""Drawing of one set-point schematic of the farm-control set-point figure (plot_joint_setpoints.py): the 5 x 1 column of set points
(tiled over the 5 columns of the farm) in top view."""

import numpy as np

SPACING = 6.0
FS = 8  # font size of the set-point text

OPT_TEXT = "\n".join([r"$\lambda_\mathrm{opt}$", r"$\theta_{p,\mathrm{opt}}$", r"$\gamma_\mathrm{opt}$", r"$C_{T,\mathrm{opt}}'$"])


def draw_setpoint_column(ax, df, color, all_opt=False, show_inflow=True):
    """Top view of one 5 x 1 column of set points (df: the rows of a tiled optimiser CSV sorted by `row`, columns tsr, pitch_deg, yaw_deg,
    Ctprime). Each rotor is a line of length D (to scale) in `color`, rotated by its yaw (negative yaw = clockwise from above, the wake
    deflects toward +y); the light dashed line is the unyawed disk. Under each rotor: lambda (1 decimal), theta_p (2), gamma (0, deg) and
    C_T' (1). The last turbine of a column (and every turbine if `all_opt`, the No-control case) is at its individually power-maximising
    set point, so its numbers are replaced by the symbols lambda_opt, theta_p,opt, gamma_opt, C_T',opt. Flow is left to right."""
    for _, r in df.iterrows():
        x = SPACING * r["row"]
        half = 0.5 * np.array([-np.sin(np.radians(r["yaw_deg"])), np.cos(np.radians(r["yaw_deg"]))])
        ax.plot([x - half[0], x + half[0]], [-half[1], half[1]], color=color, lw=2.5, solid_capstyle="butt", zorder=3)
        ax.plot([x, x], [-0.62, 0.62], ls="--", color="0.72", lw=0.9, zorder=4)  # yaw = 0, over the rotor
        if all_opt or r["row"] == len(df) - 1:
            txt = OPT_TEXT
        else:
            txt = "\n".join([rf"$\lambda = {r['tsr']:.1f}$", rf"$\theta_p = {r['pitch_deg']:+.2f}^\circ$",
                             rf"$\gamma = {int(round(r['yaw_deg']))}^\circ$", rf"$C_T' = {r['Ctprime']:.1f}$"])
        ax.text(x, -0.75, txt, ha="center", va="top", fontsize=FS, linespacing=1.25, color="k")
    ax.set_xlim(-4.0, 28.5)
    ax.set_ylim(-5.0, 0.9)
    ax.set_aspect("equal")
    ax.set_yticks([])  # y axis: the left spine only
    ax.set_xticks(SPACING * np.arange(5))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if show_inflow:
        ax.annotate("", xy=(-1.0, 0.0), xytext=(-3.0, 0.0), arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.0, mutation_scale=7))
        ax.text(-2.0, 0.25, r"$u_\infty$", ha="center", va="bottom", fontsize=FS - 1, color="k")
