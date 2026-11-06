"""
Five-panel two-turbine C_{P,farm} contour figure: LES, Gaussian wake + Cosine
rotor (cos^1.98, calibrated to C_T' = 2), Gaussian wake + UMM, Vortex wake + UMM,
and TANDEM wake + UMM, as a function of leading-turbine thrust coefficient
C_T' and yaw misalignment gamma_1. The bottom-left cell of the 2 x 3 grid is left
empty.

Each panel is colored by the percent gain in C_{P,farm} relative to that
same panel's own no-control baseline (C_T'=2.0, yaw=0deg) rather than by
the raw C_{P,farm} value, so that panels with different absolute C_{P,farm}
levels (LES vs. each reduced-order model) are directly comparable in terms
of the control benefit they predict.

Model panels (b-e) are contoured on the dense (twice-LES-resolution) grid
for a smoother contour and a more precisely-localized optimum; LES itself
is shown on its own sparse, 110-point (ctp, yaw) sample grid.

Markers: red squares are each model's own optimum; sky-blue diamonds (on every panel)
are all LES set points whose farm power is statistically indistinguishable from the
best LES set point (two-sided Welch t-test on the 10-minute window means, p > 0.05;
see `tmp/extra/twoturbine_add_les_std.py`). The small "MAE" in each model panel is the
mean absolute difference between that model's and the LES percent-gain contour at the 110
LES sample points (percentage points). The legend and the (horizontal) colorbar sit in the
empty bottom-left cell.

LES data: `LES_DATA/twoturbine_alldata_LES_std.csv`; sparse and dense model sweeps:
`R1/generate/generate_two_turbine_contours.py` (cosine Pp = 1.98).

Kirby Heck
2026
"""

import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats
from scipy.interpolate import RegularGridInterpolator
from WES2024.R1 import figuresettings as fs
from WES2024.R1.generate import generate_two_turbine_contours as sweeps
from WES2024.R1.les_data import TWOTURBINE_LES as DATA_LES  # with the 10-min window standard deviations (Cp_*_std) and window count (n_win)

# panel -> (row, col) in the 2 x 3 grid; (1, 0) is intentionally left blank
PANEL_POSITIONS = {
    "LES": (0, 0),
    "cosine_vargauss": (0, 1),
    "unified_vargauss": (0, 2),
    "unified_vortex": (1, 1),
    "unified_tandem": (1, 2),
}

DISPLAY_NAMES = {
    "LES": "LES",
    "cosine_vargauss": "Gaussian + Cosine",
    "unified_vargauss": "Gaussian + UMM",
    "unified_vortex": "Vortex + UMM",
    "unified_tandem": "TANDEM + UMM",
}

# no-control baseline set point (greedy/aligned operation)
BASELINE_CTP = 2.0
BASELINE_YAW = 0.0

CONTOUR_KW = dict(vmin=-25, vmax=25, extend="both", levels=np.arange(-25, 25.01, 2.5), cmap="PRGn")
OPT_KW = dict(s=40, lw=1, edgecolor="w", zorder=5)
LES_OPT_COLOR = "#A5D6FF"
LES_OPT_KW = dict(s=30, marker="D", color=LES_OPT_COLOR, lw=0.5, alpha=0.8, zorder=3)
ALPHA_SIGNIFICANCE = 0.05


def _load(path):
    """Loads a two-turbine sweep CSV, standardizing the farm-power column to `Cp`."""
    df = pl.read_csv(path)
    if "Cp" not in df.columns and "Cp_T" in df.columns:
        df = df.rename({c: c.replace("Cp_T", "Cp") for c in df.columns if c.startswith("Cp_T")})
    return df.sort(["ctp", "yaw"])


def baseline_cp(df, key="Cp", ctp=BASELINE_CTP, yaw=BASELINE_YAW):
    """Bilinearly-interpolated `key` at the no-control (ctp, yaw) baseline, on df's own (ctp, yaw) grid."""
    ctp_vals = np.sort(df["ctp"].unique().to_numpy())
    yaw_vals = np.sort(df["yaw"].unique().to_numpy())
    grid = df[key].to_numpy().reshape((len(ctp_vals), len(yaw_vals)))
    interp = RegularGridInterpolator((ctp_vals, yaw_vals), grid, bounds_error=False, fill_value=np.nan)
    return float(interp([(ctp, yaw)])[0])


def percent_gain(df, key="Cp"):
    """Adds a `gain` column: percent change in `key` relative to df's own no-control baseline."""
    base = baseline_cp(df, key=key)
    return df.with_columns(((pl.col(key) - base) / base * 100).alias("gain"))


def plot_contour(df, ax, key="gain", **plt_kwargs):
    """Contourf of `key` over the (yaw, ctp) sample grid in `df` (assumed to tile a regular grid)."""
    nctp = len(df["ctp"].unique())
    nyaw = len(df["yaw"].unique())
    yawG = df["yaw"].to_numpy().reshape((nctp, nyaw))
    ctpG = df["ctp"].to_numpy().reshape((nctp, nyaw))
    valG = df[key].to_numpy().reshape((nctp, nyaw))

    kwargs = {**CONTOUR_KW, **plt_kwargs}
    return ax.contourf(yawG, ctpG, valG, **kwargs)


def _dilate_mask(mask):
    """4-neighbor dilation of a 2D boolean grid, zero-padded at borders (no wraparound)."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    dilated = mask.copy()
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        dilated |= padded[1 + di : 1 + di + mask.shape[0], 1 + dj : 1 + dj + mask.shape[1]]
    return dilated


def optimum(df, key="Cp", exclude_near_nan=True):
    """
    (yaw, ctp) of the sample point maximizing `key` in `df`.

    `exclude_near_nan` (default True) additionally masks out any grid point
    within one grid step of a NaN cell before taking the max. This matters
    for the vortex-wake rotor closures, whose optimum could otherwise land at the edge
    of a small divide-by-zero NaN region (near-total-blockage breakdown of
    the underlying 1D-momentum rotor closure) rather than at the physically
    meaningful interior local max.
    """
    nctp = len(df["ctp"].unique())
    nyaw = len(df["yaw"].unique())
    yawG = df["yaw"].to_numpy().reshape((nctp, nyaw))
    ctpG = df["ctp"].to_numpy().reshape((nctp, nyaw))
    valG = df[key].to_numpy().reshape((nctp, nyaw))

    valid = ~np.isnan(valG)
    if exclude_near_nan:
        valid &= ~_dilate_mask(np.isnan(valG))

    idx = np.unravel_index(np.argmax(np.where(valid, valG, -np.inf)), valG.shape)
    return yawG[idx], ctpG[idx]


def indistinguishable_optima(les_df, key="Cp", alpha=ALPHA_SIGNIFICANCE):
    """
    LES set points whose `key` cannot be distinguished from the best (max `key`) set point.

    Two-sided Welch t-test of each point against the maximum, using the standard deviation and
    number of the 10-minute window means (`{key}_std`, `n_win`); points with p > `alpha` are kept.
    The maximum itself is always included. Points without power statistics (null std) are dropped.
    """
    best = les_df.filter(pl.col(key) == les_df[key].max()).head(1)
    mean, std, n = (best[c].item() for c in (key, f"{key}_std", "n_win"))
    keep = []
    for row in les_df.iter_rows(named=True):
        if any(row[c] is None for c in (key, f"{key}_std", "n_win")):
            keep.append(False)
            continue
        res = stats.ttest_ind_from_stats(
            mean, std, n, row[key], row[f"{key}_std"], row["n_win"], equal_var=False, alternative="two-sided"
        )
        keep.append(bool(res.pvalue > alpha))
    return les_df.filter(pl.Series(keep))


def mae(df, les_df, key="gain"):
    """NaN-safe mean absolute error of `key` between a model sweep and LES; both sit on the same 110 LES (ctp, yaw) points."""
    df, les_df = df.sort(["ctp", "yaw"]), les_df.sort(["ctp", "yaw"])
    assert np.allclose(df["ctp"], les_df["ctp"]) and np.allclose(df["yaw"], les_df["yaw"]), "sweeps are not aligned"
    err = df[key].to_numpy() - les_df[key].to_numpy()
    return float(np.nanmean(np.abs(err)))


def interp_gain(df, yaw, ctp, key="gain"):
    """Bilinearly-interpolated `key` of a regular (ctp, yaw) sweep at the set point (yaw, ctp)."""
    ctp_vals = np.sort(df["ctp"].unique().to_numpy())
    yaw_vals = np.sort(df["yaw"].unique().to_numpy())
    grid = df.sort(["ctp", "yaw"])[key].to_numpy().reshape((len(ctp_vals), len(yaw_vals)))
    interp = RegularGridInterpolator((ctp_vals, yaw_vals), grid, bounds_error=False, fill_value=np.nan)
    return float(interp([(ctp, yaw)])[0])


def print_realized_gain(data, dense):
    """
    Power gain an LES run at each model's optimal set point would realize, versus the best LES set point.

    The model optimum is the one marked in the figure (dense grid); the LES gain there is bilinearly interpolated
    from the 110-point LES sweep, relative to the LES no-control baseline.
    """
    les_df = data["LES"]
    best_yaw, best_ctp = optimum(les_df)
    les_best = interp_gain(les_df, best_yaw, best_ctp)

    print("\nRealized LES power gain at the model-optimal set points (percent change from LES no control)")
    print(f"{'':<18}{'yaw (deg)':>10}{'Ct_1':>7}{'model gain':>12}{'LES gain':>10}{'vs LES opt':>12}")
    for key in sweeps.KEYS:
        yaw, ctp = optimum(dense[key])
        realized = interp_gain(les_df, yaw, ctp)
        model_gain = interp_gain(dense[key], yaw, ctp)
        print(f"{DISPLAY_NAMES[key]:<18}{yaw:>10.1f}{ctp:>7.2f}{model_gain:>11.1f}%{realized:>9.1f}%{realized - les_best:>+11.1f}%")
    print(f"{'LES optimum':<18}{best_yaw:>10.1f}{best_ctp:>7.2f}{'':>12}{les_best:>9.1f}%")


def make_plots():
    data = {"LES": percent_gain(_load(DATA_LES)), **{key: percent_gain(_load(sweeps.cache_path(key))) for key in sweeps.KEYS}}
    dense = {key: percent_gain(_load(sweeps.cache_path(key, dense=True))) for key in sweeps.KEYS}

    les_df = data["LES"]
    les_opts = indistinguishable_optima(les_df)

    # Fixed layout (inches): square panels with uniform gaps. Automatic layouts reserve room for tick labels in every
    # row/column; here (a)'s x-label and (d)'s y-label instead hang into the empty legend cell.
    side, gap_x, gap_y = 1.65, 0.12, 0.30  # panel side; column gap; row gap (holds the titles)
    left, right, bottom, top = 0.50, 0.05, 0.50, 0.28
    width = left + 3 * side + 2 * gap_x + right
    height = bottom + 2 * side + gap_y + top
    fig, axarr = plt.subplots(nrows=2, ncols=3, figsize=(width, height), sharex=True, sharey=True)
    for (row, col), ax in np.ndenumerate(axarr):
        ax.set_box_aspect(1)  # square panels
        x0 = left + col * (side + gap_x)
        y0 = bottom + (1 - row) * (side + gap_y)
        ax.set_position([x0 / width, y0 / height, side / width, side / height])
    key_ax = axarr[1, 0]  # empty cell: holds the legend and colorbar
    key_ax.axis("off")
    im = None
    for k, (key, (row, col)) in enumerate(PANEL_POSITIONS.items()):
        ax = axarr[row, col]
        ax.set_title(f"({chr(97 + k)}) {DISPLAY_NAMES[key]}", fontsize=9, loc="left")

        if key == "LES":
            im = plot_contour(les_df, ax)
            ax.scatter(les_df["yaw"], les_df["ctp"], s=5, marker="o", color="w", edgecolor="k", lw=0.5, zorder=4)
        else:
            im = plot_contour(dense[key], ax)
            model_yaw, model_ctp = optimum(dense[key])
            ax.scatter(model_yaw, model_ctp, c="tab:red", marker="s", **OPT_KW)

            # error on the sparse, LES-aligned grid (the dense grid has no LES to compare against)
            ax.text(
                0.97, 0.03, rf"MAE: {mae(data[key], les_df):.1f}\%", fontsize=10, ha="right", va="bottom",
                transform=ax.transAxes, zorder=6, bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="none", alpha=0.8),
            )

        ax.scatter(les_opts["yaw"], les_opts["ctp"], **LES_OPT_KW)

    # LES sits above the blank cell, so it needs its own x-axis labelling; panel (d) sits right of that cell, so it
    # needs its own y-axis labelling
    axarr[0, 0].tick_params(labelbottom=True)
    axarr[1, 1].tick_params(labelleft=True)
    for ax in (axarr[0, 0], axarr[1, 1], axarr[1, 2]):
        ax.set_xlabel(r"$\gamma_1$ (deg.)")
    for ax in (axarr[0, 0], axarr[1, 1]):
        ax.set_ylabel("$C_{T,1}'$")

    key_ax.legend(
        handles=[
            Line2D([], [], ls="", marker="D", markersize=4.5, color=LES_OPT_COLOR, alpha=0.8),
            Line2D([], [], ls="", marker="s", markersize=6, markerfacecolor="tab:red", markeredgecolor="w", markeredgewidth=1),
        ],
        labels=["Statistical LES optima", "Model optimum"],
        loc="upper left", bbox_to_anchor=(-0.30, 0.62), frameon=False, fontsize=8, handletextpad=0.4,
    )
    cax = key_ax.inset_axes([-0.2, 0.2, 0.9, 0.06])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=np.arange(-20, 21, 10))
    cb.set_label(r"$\Delta P_\mathrm{farm}$ \%", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fs.save()
    plt.close()

    print_realized_gain(data, dense)


if __name__ == "__main__":
    """Make plots"""
    make_plots()
