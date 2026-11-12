# dualitic is imported first to ensure correct monkey patching.
import dualitic

"""
R1: one yawed BEM rotor in uniform flow under the three yaw strategies of the paper, as a function of yaw.

- Yaw:       pitch and TSR fixed at the single-turbine optimum (YawControlBEM).
- Yaw (mod): pitch fixed, TSR = TSR_OPT cos(yaw), i.e. the modified tip-speed ratio lambda~ = Omega R / (U cos yaw) is
             held constant (YawControlModBEM; Li & Yang, JFM 2021, who recover cos^2 thrust and cos^3 power loss).
- Yaw (KO):  pitch fixed, TSR from the K-omega^2 torque law P ~ omega^3 (YawKOmegaControlBEM, R1 normalisation
             Cp / REWS^3 / tsr^3 = CP_OPT / TSR_OPT^3).
- Joint:     pitch and TSR chosen to maximise Cp at each yaw (same bounds as JointControlBEM). For an isolated rotor
             this is the best that any pitch/TSR choice can do, so it is the upper bound on Cp.

Results are cached in data/generate_single_turbine_yaw_strategies.csv. Usage: python -m WES2024.R1.generate.generate_single_turbine_yaw_strategies
"""

from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm
from mitwindfarm import Layout
from scipy.optimize import brentq, minimize

from WES2024 import utils
from WES2024.CustomRotors import BEMUnifiedMomentumLUTAnnulus
from WES2024.R1.wakemodels import get_bem_lut_rotor, get_windfarm

PITCH_OPT = BEMUnifiedMomentumLUTAnnulus.PITCH_OPT
TSR_OPT = BEMUnifiedMomentumLUTAnnulus.TSR_OPT
CP_OPT = BEMUnifiedMomentumLUTAnnulus.CP_OPT

FILESTEM = Path(__file__).stem
CSV = utils.CACHEDIR / f"{FILESTEM}.csv"
YAWS = np.arange(0.0, 45.1, 1.0)  # [deg]; 45 deg is the bound used by the BEM controllers
PITCH_BOUNDS = (-np.deg2rad(10), np.deg2rad(10))  # as JointControlBEM
TSR_BOUNDS = (3.0, 10.0)

_windfarm = None


def _real(v):
    return float(np.ravel(getattr(v, "real", v))[0])


def _rotor(pitch, tsr, yaw):
    global _windfarm
    if _windfarm is None:
        _windfarm = get_windfarm("varvortex", rotor_model=get_bem_lut_rotor())  # one rotor: the wake model is irrelevant
    return _windfarm(Layout([0.0], [0.0]), [(pitch, tsr, yaw)]).rotors[0]


def _row(method, yaw, pitch, tsr):
    r = _rotor(pitch, tsr, yaw)
    return dict(method=method, yaw=np.rad2deg(yaw), pitch=np.rad2deg(pitch), tsr=tsr,
                Cp=_real(r.Cp), Ct=_real(r.Ct), Ctprime=_real(r.Ctprime), an=_real(r.an), u4=_real(r.u4))


def _komega_tsr(yaw):
    K0 = CP_OPT / TSR_OPT**3

    def residual(tsr):
        r = _rotor(PITCH_OPT, tsr, yaw)
        return _real(r.Cp) / _real(r.REWS) ** 3 / _real(r.extra.tsr) ** 3 - K0

    return brentq(residual, 3.0, 12.0, xtol=1e-10)


def _joint(yaw, guess):
    """Cp-optimal (pitch, tsr) at fixed yaw. Cp is a very flat ridge in (pitch, tsr) at large yaw (changes of ~1e-4
    over several degrees of pitch), so gradient-based solvers stall along it: use a tight Nelder-Mead from several
    starts (previous yaw, unyawed optimum, high-yaw optimum) in (pitch [deg], tsr) and keep the best."""

    def neg_cp(x):
        if not (np.rad2deg(PITCH_BOUNDS[0]) <= x[0] <= np.rad2deg(PITCH_BOUNDS[1]) and TSR_BOUNDS[0] <= x[1] <= TSR_BOUNDS[1]):
            return 1.0
        return -_real(_rotor(np.deg2rad(x[0]), x[1], yaw).Cp)

    starts = [np.array([np.rad2deg(guess[0]), guess[1]]), np.array([np.rad2deg(PITCH_OPT), TSR_OPT]), np.array([-5.0, 8.1])]
    best = None
    for x0 in starts:
        res = minimize(neg_cp, x0, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-12, maxiter=600,
                                                                      initial_simplex=x0 + np.array([[0, 0], [0.5, 0], [0, 0.2]])))
        if best is None or res.fun < best.fun:
            best = res
    return np.deg2rad(best.x[0]), best.x[1]


def generate(regenerate=False):
    if CSV.exists() and not regenerate:
        return pl.read_csv(CSV)

    rows, guess = [], (PITCH_OPT, TSR_OPT)
    for yd in tqdm(YAWS, desc="Generating data"):
        yaw = np.deg2rad(yd)
        rows.append(_row("Yaw", yaw, PITCH_OPT, TSR_OPT))
        rows.append(_row("YawMod", yaw, PITCH_OPT, TSR_OPT * np.cos(yaw)))
        rows.append(_row("YawKOmega", yaw, PITCH_OPT, _komega_tsr(yaw)))
        guess = _joint(yaw, guess)  # warm start from the previous yaw
        rows.append(_row("Joint", yaw, *guess))
    df = pl.DataFrame(rows)
    df.write_csv(CSV)
    return df


if __name__ == "__main__":
    print(generate(regenerate=True))
