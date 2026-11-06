#!/usr/bin/python

# Hannah Johlas, 2019
# Revised Kirby Heck, 2026

import inspect
import matplotlib
import matplotlib.pyplot
from pathlib import Path

# Plot formatting
matplotlib.rcParams['savefig.format']     = 'png'                               # image file type (pdf, png)
matplotlib.rcParams['savefig.dpi']        = 600                                 # figure resolution (makes a little better .png)
matplotlib.rcParams['savefig.pad_inches'] = 0.10                                # remove extra whitespace
matplotlib.rcParams['savefig.bbox']       = 'tight'                           # remove extra whitespace = 'tight'
matplotlib.rcParams['lines.linewidth']    = 1.
matplotlib.rcParams['legend.handlelength']  = 1.5
matplotlib.rcParams['font.family']        = 'serif'
matplotlib.rcParams['axes.linewidth'] = 0.5 # set the value globally
matplotlib.rcParams['xtick.major.width'] = 0.5
matplotlib.rcParams['ytick.major.width'] = 0.5
matplotlib.rcParams['text.usetex']           = True
matplotlib.rcParams['figure.figsize']        = [4,3]
matplotlib.rcParams['figure.dpi']            = 150

BASE = Path(__file__).parent
FIGPATH = BASE / "figs"
FIGPATH.mkdir(exist_ok=True, parents=True)


def save(stem=None, **kwargs):
    """Save the current figure to the FIGPATH directory with a filename based on the caller's name."""
    if stem is None:
        caller_frame = inspect.stack()[1]
        stem = Path(caller_frame.filename).stem
    # append the extension explicitly: a "." in the stem (e.g. "..._Cr0.2") must not be read as one
    fname = FIGPATH / f"{stem}.{matplotlib.rcParams['savefig.format']}"
    matplotlib.pyplot.savefig(fname, **kwargs)
    print(f"Saved figure to {fname}")


# ---------------------------------------------------------------------------
# include shared color maps below:

# Colorblind-safe palette (seaborn/Okabe-Ito "colorblind") and matching linestyles for the six control strategies
# used across the R1 figures, so each line is identifiable by color AND dash pattern alone (e.g. in grayscale
# print or for colorblind readers). Scoped to R1/forpaper; the legacy WES2024.utils.controller_colors (tab10-based)
# is left untouched so older, already-finalised figures elsewhere in the repo don't shift color.
import seaborn as sns

_CB = sns.color_palette("colorblind").as_hex()

CONTROLLER_COLORS = {
    "NoControl": "k",
    "ThrustControl": _CB[0],        # blue
    "YawControl": _CB[1],           # orange
    "YawModControl": _CB[4],        # purple
    "YawKOmegaControl": _CB[3],     # vermillion
    "JointControl": _CB[2],         # green
}

CONTROLLER_LINESTYLES = {
    "NoControl": "-",
    "ThrustControl": (0, (6, 2)),
    "YawControl": (0, (1, 1)),
    "YawModControl": (0, (4, 1, 1, 1)),
    "YawKOmegaControl": (0, (4, 1, 1, 1, 1, 1)),
    "JointControl": (0, (8, 2)),
}


# Legend labels for the six control strategies, shared by the single-turbine yaw-strategies figure and the two-turbine
# figures. The yaw strategies are named by what is held fixed ("Yaw (theta_p,opt, lambda_opt)": pitch and TSR at the
# single-turbine optimum; "(theta_p,opt, lambda~)": constant modified TSR; "(K Omega^2)": TSR follows the torque law).
controller_labels = {
    "NoControl": "No Control",
    "ThrustControl": "Thrust Control",
    "YawModControl": r"Yaw ($\theta_{p,\mathrm{opt}}$, $\tilde{\lambda}$)",
    "YawKOmegaControl": r"Yaw ($K\Omega^2$)",
    "YawControl": r"Yaw ($\theta_{p,\mathrm{opt}}$, $\lambda_\mathrm{opt}$)",
    "JointControl": "Joint Control",
}

# ---------------------------------------------------------------------------
# Farm-control study (R1/forpaper): names, colors, hatching and markers shared by the set-point, LES-gain and row-wise power figures.
# Wake model = the model whose optimiser produced the set points (and, for the model-power bars, the model that predicts the power);
# bar colour identifies the model, bar hatch the control strategy.
matplotlib.rcParams['hatch.linewidth'] = 0.8

MODEL_LABELS = {"TANDEM": "TANDEM", "Vortex": "Vortex sheet", "Gauss": "Gaussian"}
MODEL_COLORS = {"TANDEM": "#0072B2", "Vortex": "#E69F00", "Gauss": "#009E73"}
NC_COLOR = "0.85"  # No-control bars

CONTROL_LABELS = {"NoControl": "No control", "Thrust": "Thrust", "Yaw": r"Yaw $K\Omega^2$", "YawMod": r"Yaw ($\tilde{\lambda}$)", "Joint": "Joint"}  # as named in the other sections
CONTROL_HATCH = {"NoControl": None, "Thrust": "..", "Yaw": "OO", "YawMod": "xx", "Joint": "//"}

# row-wise power figures: markers distinguish the wake models, LES is the dark squares
VARIANT_LABELS = MODEL_LABELS
VARIANT_COLORS = MODEL_COLORS
VARIANT_MARKERS = {"TANDEM": "o", "Vortex": "s", "Gauss": "^"}
LES_COLOR = "0.15"
