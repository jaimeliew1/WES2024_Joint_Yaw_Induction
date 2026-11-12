# dualitic is imported first to ensure correct monkey patching.
import dualitic

"""
R1: row-wise power of the wake models for the farm-control set points, cached. A 5x1 column (rows 1-5, 6 D apart, wdir 0) is solved
for every set of set points of farm_control_cases.CASES with every wake model on the SAME rotor model, so that rotor and wake
modelling can be told apart in the comparison with the LES (forpaper/plot_rowwise_power.py):

    wake models ("variants")   TANDEM (with the ground),
                               Vortex sheet (varvortex), Gaussian (gaussBP); uniform inflow, TIamb = wakemodels.TIAMB.
    rotor models               "umm":       the annulus BEM-LUT rotor (BEMLUT), fed (pitch, TSR, yaw) of the optimiser CSV,
                               "tabulated": TabulatedLESAD (Cp, Ct of the leading turbine of the two-turbine ABL LES sweep as functions
                                            of (Ct', |yaw|)), fed (Ct', yaw) of the optimiser CSV -- removes the rotor-model error.
Every variant is fed the set points of the control case (e.g. "Joint_Vortex": the Vortex-optimised Joint set points), not its own.

Cached in data/generate_rowwise_power_models.npz as arrays "<rotor>__<case>__<variant>" (5,) of Cp [rotor-disk power coefficient of each
row, including REWS^3] -- the plotting normalises by the same variant's No-control row-1 Cp ("<rotor>__NoControl__<variant>").

Usage: python -m WES2024.R1.generate.generate_rowwise_power_models   (~1 min)
"""

from pathlib import Path

import numpy as np
from mitwindfarm import Area

from WES2024 import utils
from WES2024.R1.farm_control_cases import CASES, case_setpoints, column_layout, model_nocontrol_setpoints
from WES2024.R1.tabulated_rotor import TabulatedLESAD
from WES2024.R1.wakemodels import get_bem_lut_rotor, get_windfarm


FILESTEM = Path(__file__).stem
LAYOUT = column_layout()
VARIANTS = {"TANDEM": "tandem", "Vortex": "varvortex", "Gauss": "gaussBP"}  # variant -> get_windfarm key
ROTORS = ("umm", "tabulated")


def make_rotor(kind):
    if kind == "umm":
        return get_bem_lut_rotor(rotor_grid=Area(), show_warnings=False)
    return TabulatedLESAD()


def cp_rows(variant, rotor, sp):
    """Cp (5,) of the 5x1 column at the set-point frame `sp` (rows of an optimiser CSV)."""
    cols = getattr(rotor, "setpoint_columns", ("pitch", "tsr", "yaw"))
    wf = get_windfarm(VARIANTS[variant], rotor_model=rotor)
    return np.array([r.Cp for r in wf(LAYOUT, list(zip(*(sp[c] for c in cols)))).rotors])


def generate(regenerate=False):
    f = utils.CACHEDIR / f"{FILESTEM}.npz"
    if f.exists() and not regenerate:
        d = np.load(f)
        return {k: d[k] for k in d}
    out = {}
    for rotor_kind in ROTORS:
        rotor = make_rotor(rotor_kind)
        for key in CASES:
            # the No-control set points are the same for every model; every other case is read from its own optimiser CSV
            sp = model_nocontrol_setpoints("TANDEM") if key == "NoControl" else case_setpoints(key)
            for v in VARIANTS:
                out[f"{rotor_kind}__{key}__{v}"] = cp_rows(v, rotor, sp)
    return out


if __name__ == "__main__":
    d = generate(regenerate=True)
    np.savez_compressed(utils.CACHEDIR / f"{FILESTEM}.npz", **d)
    print(f"Saved {utils.CACHEDIR / (FILESTEM + '.npz')}: {len(d)} arrays")
