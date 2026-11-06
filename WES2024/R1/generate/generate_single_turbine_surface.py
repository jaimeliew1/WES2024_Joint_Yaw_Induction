# dualitic is imported first to ensure correct monkey patching.
import dualitic

"""
R1: Cp/Ct surface over (pitch, TSR) at two yaw angles, with the Ct'-minimising derate trajectory overlaid -- same
figure as WES2024/Plot/ForPaper/Fig00_single_turbine_surface.py (built from Generate/pitch_tsr_surface.py and
Generate/minCt_trajectory.py), but evaluated through the R1 annulus-closure BEM LUT (wakemodels.get_bem_lut_rotor)
instead of a live per-point BEM solve. The table spline is ~3 orders of magnitude cheaper than a BEM solve (see
check at the bottom of this docstring), so the (pitch, tsr) grid and the isobar root-finds for the derate
trajectory are evaluated directly and serially -- no foreach/multiprocessing needed.

The Ct'-minimising derate-trajectory algorithm (find_optimal_setpoint, Cp_isobars, Ctprime_minimising_Cp_setpoint,
generate_derate_strat) is reused unmodified from WES2024/Generate/minCt_trajectory.py: those functions only need a
callable `bem(pitch, tsr, yaw)` returning an object with .Cp() and .Ctprime(), which _lut_bem below provides.

Results are cached in data/generate_single_turbine_surface_{surface,trajectory}.csv.
Usage: python -m WES2024.R1.generate.generate_single_turbine_surface
"""

from pathlib import Path

import numpy as np
import polars as pl

from WES2024 import utils
from WES2024.CustomRotors import BEMUnifiedMomentumLUTAnnulus
from WES2024.Generate.minCt_trajectory import generate_derate_strat
from WES2024.R1.wakemodels import get_bem_lut_rotor

PITCH_OPT = BEMUnifiedMomentumLUTAnnulus.PITCH_OPT
TSR_OPT = BEMUnifiedMomentumLUTAnnulus.TSR_OPT
CP_OPT = BEMUnifiedMomentumLUTAnnulus.CP_OPT

FILESTEM = Path(__file__).stem
SURFACE_CSV = utils.CACHEDIR / f"{FILESTEM}_surface.csv"
TRAJECTORY_CSV = utils.CACHEDIR / f"{FILESTEM}_trajectory.csv"

PITCHES = np.deg2rad(np.linspace(-15, 15, 50))  # [rad]
TSRS = np.linspace(5, 10.5, 50)
YAWS = [0.0, 45.0]  # [deg], matching Fig00_single_turbine_surface.YAW2

_table = None


def _get_table():
    global _table
    if _table is None:
        _table = get_bem_lut_rotor().table  # annulus-closure LUT; generated on first use (no-op when cached)
    return _table


class _LUTPoint:
    """Stand-in for a MITRotor BEMSolution backed by the R1 annulus-closure BEM LUT, so
    WES2024.Generate.minCt_trajectory's Ct'-minimising derate-trajectory algorithm can be reused unmodified."""

    def __init__(self, pitch, tsr, yaw):
        self.pitch, self.tsr, self.yaw = pitch, tsr, yaw
        self._t = _get_table()(pitch, tsr, yaw)

    def Cp(self):
        return float(self._t["Cp"])

    def Ctprime(self):
        return float(self._t["Ctprime"])


def _lut_bem(pitch, tsr, yaw):
    return _LUTPoint(pitch, tsr, yaw)


def _generate_surface():
    rows = []
    for yd in YAWS:
        yaw = np.deg2rad(yd)
        for p in PITCHES:
            for tsr in TSRS:
                t = _get_table()(p, tsr, yaw)
                rows.append(dict(pitch=np.rad2deg(p), tsr=tsr, yaw=yd, Cp=t["Cp"], Ct=t["Ct"]))
    return pl.DataFrame(rows)


def _generate_trajectory():
    dfs = []
    for yd in YAWS:
        yaw = np.deg2rad(yd)
        df = generate_derate_strat(_lut_bem, yaw, N_Cp=20).with_columns(pl.col("pitch").degrees(), yaw=pl.lit(yd))
        dfs.append(df)
    return pl.concat(dfs)


def generate(regenerate=False):
    if SURFACE_CSV.exists() and TRAJECTORY_CSV.exists() and not regenerate:
        return pl.read_csv(SURFACE_CSV), pl.read_csv(TRAJECTORY_CSV)
    df_surface = _generate_surface()
    df_trajectory = _generate_trajectory()
    df_surface.write_csv(SURFACE_CSV)
    df_trajectory.write_csv(TRAJECTORY_CSV)
    return df_surface, df_trajectory


if __name__ == "__main__":
    df_surface, df_trajectory = generate(regenerate=True)
    print(df_surface)
    print(df_trajectory)
