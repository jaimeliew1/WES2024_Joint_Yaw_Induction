# dualitic is imported first to ensure correct monkey patching.
import dualitic

"""
Look-up table of the IEA 15 MW BEM rotor in *uniform* inflow: (pitch, TSR, yaw) -> (Cp, Ct, Ct', an, u4, v4, x0),
all normalised by the rotor-effective wind speed (REWS), exactly as the BEM rotor reports them for U/REWS = 1.

Same purpose as tmp/extra/generate_bem_lut.py (feeds WES2024.CustomRotors.BEMLUT via wakemodels.get_bem_lut_rotor), but
built from get_bem_annulus_rotor: annulus-averaged, standard local tip-loss correction Cx_corr = Cx/F (clipped to
Ct <= 1.5) closed with the Unified Momentum Model's LUT, instead of get_bem_rotor's disk-integrated-then-redistributed
"rotor_induction_tiploss" averaging. Writes to a distinct cache file (prefix "bem_lut_annulus_") so the original
table/grid is never touched. Nr=128, Ntheta=12 chosen from R1/check convergence sweeps: <=0.22% Cp residual vs
Nr=180 (and Ntheta=12 already converged to <0.02% against Ntheta=40) across this bounds' corners.

What is neglected by tabulating at uniform inflow: shear, veer and wake deficits *across the disk* (partial wakes);
check with tmp/extra/check_bem_lut.py.

Usage: python -m WES2024.R1.generate.generate_bem_lut_annulus [--regenerate]
  env NR (128), DPITCH [deg] (0.5), DTSR (0.25), DYAW [deg] (3.0), PROCESSES (96)
  Grid: pitch -15..15 deg, TSR 4..12, yaw -45..45 deg.
  Cost of the default grid (62403 BEM solves): ~40 core-hours, e.g. 25 min on 96 workers (measured: 1486 s) at ~0.2 GB per worker.
"""

import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from foreach import foreach
from MITRotor import BEMGeometry

from WES2024 import utils
from WES2024.R1.wakemodels import get_bem_annulus_rotor

__all__ = ["lut_path", "generate", "PITCH_RANGE", "TSR_RANGE", "YAW_RANGE"]

NR = int(os.environ.get("NR", 128))
NTHETA = 12
DPITCH = float(os.environ.get("DPITCH", 0.5))
DTSR = float(os.environ.get("DTSR", 0.25))
DYAW = float(os.environ.get("DYAW", 3.0))
PROCESSES = int(os.environ.get("PROCESSES", min(96, os.cpu_count())))  # one BEM solve holds ~0.2 GB per worker

PITCH_RANGE = (-15.0, 15.0)  # deg
TSR_RANGE = (4.0, 12.0)
YAW_RANGE = (-45.0, 45.0)  # deg


def lut_path(nr=NR, dpitch=DPITCH, dtsr=DTSR, dyaw=DYAW) -> Path:
    return utils.CACHEDIR / f"bem_lut_annulus_Nr{nr}_p{dpitch:g}_t{dtsr:g}_y{dyaw:g}.csv"


def _axes(dpitch, dtsr, dyaw):
    return (
        np.linspace(*PITCH_RANGE, int(round((PITCH_RANGE[1] - PITCH_RANGE[0]) / dpitch)) + 1),
        np.linspace(*TSR_RANGE, int(round((TSR_RANGE[1] - TSR_RANGE[0]) / dtsr)) + 1),
        np.linspace(*YAW_RANGE, int(round((YAW_RANGE[1] - YAW_RANGE[0]) / dyaw)) + 1),
    )


_rotor = {}


def _solve_point(x):
    """One BEM solve at (nr, pitch [deg], tsr, yaw [deg]); one foreach task per set point so the progress bar counts solves."""
    nr, pitch, tsr, yaw = x
    if nr not in _rotor:
        m = get_bem_annulus_rotor(BEMGeometry(Nr=nr, Ntheta=NTHETA))._model
        xg, _, _ = m.sample_points()
        _rotor[nr] = (m, np.ones_like(xg), np.zeros_like(xg))
    m, U, wdir = _rotor[nr]
    s = m(np.deg2rad(pitch), tsr, yaw=np.deg2rad(yaw), tilt=0.0, U=U, wdir=wdir)
    return dict(
        pitch=pitch, tsr=tsr, yaw=yaw, Cp=float(s.Cp()), Ct=float(s.Ct()), Ctprime=float(s.Ctprime()),
        an=float(np.ravel(s.a())[0]), u4=float(s.u4), v4=float(s.v4), x0=float(s.x0),
        converged=bool(s.converged), niter=int(s.niter[1]),
    )


def generate(nr=NR, dpitch=DPITCH, dtsr=DTSR, dyaw=DYAW, regenerate=False) -> pl.DataFrame:
    fn = lut_path(nr, dpitch, dtsr, dyaw)
    if fn.exists() and not regenerate:
        return pl.read_csv(fn)
    pitches, tsrs, yaws = _axes(dpitch, dtsr, dyaw)
    t0 = time.time()
    rows = foreach(_solve_point, [(nr, p, t, y) for p, t, y in product(pitches, tsrs, yaws)], context="spawn",
                   parallel=True, processes=PROCESSES)
    df = pl.DataFrame(rows).sort(["pitch", "tsr", "yaw"])
    df.write_csv(fn)
    print(f"{len(df)} BEM solves in {time.time() - t0:.0f} s -> {fn}; not converged: {(~df['converged']).sum()}")
    return df


if __name__ == "__main__":
    df = generate(regenerate="--regenerate" in sys.argv)
    print(df.select("Cp", "Ct", "Ctprime", "an", "u4", "v4", "x0").describe())
