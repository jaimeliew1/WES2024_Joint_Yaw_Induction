"""
LES set points for the 5 x 5 square farm (6 D spacing both ways) aligned with the wind (wdir = 0), from 5 x 1 column optimisations
that are tiled over the 5 columns.

Why a column: for wind directions close to the farm axis the columns do not interact (tmp/extra/generate_columns_vs_farm_vortex: joint
set points within 1 % of the full 5 x 5 optimum up to ~7.5 deg for the vortex sheet). A 5 x 1 TANDEM optimisation (15 variables) is
affordable with AD; the 25-turbine one (75 variables) is not.

Cases (wake model + tabulated BEM rotor), controllers and variables per turbine
  tandemwall_bem  TANDEM with the ground        NoControl; Thrust (pitch, TSR); YawKOmega (TSR, yaw; TSR from K-omega^2, pitch fixed);
                                                YawMod (yaw only; pitch = PITCH_OPT, TSR = TSR_OPT cos(yaw)); Joint (pitch, TSR, yaw)
  vortex_bem      vortex sheet                  NoControl; Joint
  gaussBP_bem     Bastankhah & Porte-Agel Gaussian   NoControl; Joint
|yaw| <= 30 deg for every strategy (R1.controllers.YAW_MAX). Every case uses the tabulated BEM rotor (CustomRotors.BEMLUT, annulus-averaged
UMM-LUT closure, generate_bem_lut_annulus.py), never the direct DualBEM solve: set-point and Cp differences between the two were checked
to be <= 0.21 % (farm Cp, 5 x 1 column, vortex and gaussBP wakes) and the LUT is ~100x cheaper per gradient.

Optimiser: SLSQP with AD gradients on scaled variables (R1.controllers.ConstrainedMixin), from yaw = -0.05 rad. The aligned farm is exactly
left-right symmetric, so the two signs of yaw are mirror images of one solution; a single start keeps every case on the same branch. TANDEM
carries integrator noise (~1e-6 in Cp), hence ftol 1e-6 (R1.controllers.SLSQP_FTOL); the other models are noise free, ftol 1e-11.

The K-omega^2 law is Cp_local / TSR^3 = CP_REF / TSR_OPT^3 with CP_REF = the single-turbine Cp of the case's own rotor at
(PITCH_OPT, TSR_OPT), measured at start-up. Each result carries `code_hash` (WES2024 sources) and `cp_ref`, `cp_opt_stale` (CP_REF / CP_OPT - 1).

Stages (one CSV per run, so an interrupted sweep resumes):
  runs      data/<FILESTEM>/<case>/<strategy>_<start>_w000.0.csv     5 x 1 optimum, with timing
  tiled     data/<FILESTEM>/tiled/<case>_<strategy>_w000.0.csv       5 x 5 farm (turbine = column * 5 + row), 5 x 1 set points tiled, re-evaluated
Final output data/<FILESTEM>.csv concatenates the tiled files.

Usage: python -m WES2024.R1.generate.generate_les_setpoints_tiled5x5 [--regenerate]
  env PROCESSES (default 24). Run with POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1.
"""
# dualitic is imported first to ensure correct monkey patching.
import dualitic

import hashlib
import os
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
from foreach import foreach
from mitwindfarm import Area, Layout

from WES2024 import optimise, utils
from WES2024.R1 import controllers
from WES2024.R1.controllers import SLSQP_FTOL, make_controller, use_annulus_optimum
from WES2024.R1.farm_control_cases import NX, NY, column_layout, farm_layout
from WES2024.R1.wakemodels import get_bem_lut_rotor, get_windfarm

use_annulus_optimum()
PITCH_OPT, TSR_OPT, CP_OPT = optimise.PITCH_OPT, optimise.TSR_OPT, optimise.CP_OPT

__all__ = ["generate", "CASES"]

FILESTEM = Path(__file__).stem
RUNDIR = utils.CACHEDIR / FILESTEM
PROCESSES = int(os.environ.get("PROCESSES", 24))
WDIR = 0.0

TANDEM_STRATEGIES = ("Joint", "YawKOmega", "Thrust", "YawMod", "NoControl")
CASES = {
    "tandemwall_bem": dict(wake="tandem", strategies=TANDEM_STRATEGIES),
    "vortex_bem": dict(wake="varvortex", strategies=("Joint", "NoControl")),
    "gaussBP_bem": dict(wake="gaussBP", strategies=("Joint", "NoControl")),
}
METHODS = dict(Thrust="ThrustControl", YawKOmega="YawKOmegaControl", YawMod="YawModControl", Joint="JointControl", NoControl="NoControl")
YAW_START = {"Joint": -0.05, "YawKOmega": -0.05, "YawMod": -0.05}  # [rad]; Thrust and NoControl have no yaw variable
# (ftol, maxiter) of the SLSQP run per wake model
OPT = dict(tandem=(SLSQP_FTOL, 100), varvortex=(1e-11, 400), gaussBP=(1e-11, 400))

SRC_FILES = [Path(__file__).parents[2] / f for f in
             ("BEM_gradients.py", "CustomRotors.py", "CustomTangentialInduction.py", "optimise.py", "R1/wakemodels.py", "R1/controllers.py")]


def code_hash():
    return hashlib.md5(b"".join(f.read_bytes() for f in SRC_FILES)).hexdigest()[:10]


def make_windfarm(case):
    cfg = CASES[case]
    return get_windfarm(cfg["wake"], rotor_model=get_bem_lut_rotor(rotor_grid=Area()), solver_overrides=cfg.get("solver_overrides"))


@lru_cache(maxsize=None)
def cp_ref(case):
    """Single-turbine Cp of the case's BEM rotor at (PITCH_OPT, TSR_OPT), unyawed, uniform inflow (K-omega^2 reference)."""
    sol = make_windfarm(case)(Layout([0.0], [0.0]), [(PITCH_OPT, TSR_OPT, 0.0)])
    return float(sol.rotors[0].Cp)


def _start(strategy):
    return "neg" if strategy in YAW_START else "none"


def _run_file(case, strategy):
    return RUNDIR / case / f"{strategy}_{_start(strategy)}_w{WDIR:05.1f}.csv"


def _tiled_file(case, strategy):
    return RUNDIR / "tiled" / f"{case}_{strategy}_w{WDIR:05.1f}.csv"


def _write(fn, df):
    fn.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(fn)
    return fn


def _fail(fn):
    d = RUNDIR / "failed"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{fn.parent.name}_{fn.stem}.txt").write_text(traceback.format_exc())
    return None


def _run(x):
    """One 5 x 1 optimisation -> run file."""
    case, strategy = x
    fn = _run_file(case, strategy)
    if fn.exists():
        return fn
    try:
        t0 = time.time()
        controllers.CP_OPT = cp_ref(case)  # K-omega^2 reference, see module docstring
        wake = CASES[case]["wake"]
        ftol, maxiter = OPT[wake]
        controller = make_controller(METHODS[strategy], column_layout(WDIR), make_windfarm(case), wake, ftol, maxiter, yaw0=YAW_START.get(strategy))
        sol = controller.optimise(Cp_constraint=None, use_gradients=True)
        res = getattr(controller, "result", None)
        conv = [getattr(r.extra, "converged", True) for r in sol.rotors]
        return _write(fn, utils.to_polars(sol).with_columns(
            pl.lit(case).alias("case"), pl.lit(strategy).alias("strategy"), pl.lit(_start(strategy)).alias("start"),
            pl.lit(WDIR).alias("wdir"), pl.lit(float(sol.Cp)).alias("Cp_farm"), pl.lit(time.time() - t0).alias("seconds"),
            pl.lit(None if res is None else bool(res.success), dtype=pl.Boolean).alias("success"),
            pl.lit(None if res is None else int(res.nit), dtype=pl.Int64).alias("nit"),
            pl.lit(None if res is None else int(res.nfev), dtype=pl.Int64).alias("nfev"),
            pl.lit(None if res is None else str(res.message), dtype=pl.Utf8).alias("message"),
            pl.lit(bool(np.all(conv))).alias("bem_converged"),
            pl.lit(cp_ref(case)).alias("cp_ref"), pl.lit(cp_ref(case) / CP_OPT - 1).alias("cp_opt_stale"),
            pl.lit(code_hash()).alias("code_hash"),
        ))
    except BaseException:
        return _fail(fn)


def tiled_frame(sol, layout, case, strategy, note, Cp_column, start, seconds_5x1, success, nit, bem_converged, code, cp_reference, seconds_tile):
    """Output table of a 5 x 5 farm solution; the column names are read downstream (farm_control_cases, the LES input writer)."""
    n = len(layout)
    df = utils.to_polars(sol).with_columns(
        pl.Series("column", [k // NX for k in range(n)]), pl.Series("row", [k % NX for k in range(n)]),
        pl.lit(case).alias("case"), pl.lit(CASES[case]["wake"]).alias("wake_model"), pl.lit("BEM(LUT)").alias("rotor_model"),
        pl.lit(strategy).alias("strategy"), pl.lit(WDIR).alias("wdir"), pl.lit(note).alias("note"),
        pl.lit(float(sol.Cp)).alias("Cp_farm"), pl.lit(Cp_column).alias("Cp_column"),
        pl.lit(start).alias("start"), pl.lit(seconds_5x1).alias("seconds_5x1"),
        pl.lit(success, dtype=pl.Boolean).alias("success"), pl.lit(nit, dtype=pl.Int64).alias("nit"),
        pl.lit(bem_converged).alias("bem_converged"), pl.lit(code).alias("code_hash"),
        pl.lit(cp_reference).alias("cp_ref"), pl.lit(seconds_tile).alias("seconds_tile"),
    )
    # named set point columns (setpoint_j keeps the optimiser's order: pitch, TSR, yaw)
    df = df.with_columns(pl.col("setpoint_0").alias("pitch"), pl.col("setpoint_1").alias("tsr"),
                         pl.col("setpoint_2").alias("yaw_setpoint"), pl.col("setpoint_0").degrees().alias("pitch_deg"))
    return df.with_columns(pl.col("yaw").degrees().alias("yaw_deg"))


def _tile(x):
    """Tile the 5 x 1 set points over the 5 columns and re-evaluate the 5 x 5 farm."""
    case, strategy = x
    fn = _tiled_file(case, strategy)
    if fn.exists():
        return fn
    try:
        t0 = time.time()
        col = pl.read_csv(_run_file(case, strategy), infer_schema_length=None).sort("turbine")
        sp = [tuple(float(r[f"setpoint_{j}"]) for j in range(3)) for r in col.iter_rows(named=True)]
        layout = farm_layout(WDIR)
        sol = make_windfarm(case)(layout, sp * NY)  # turbine index = column * NX + position in column
        return _write(fn, tiled_frame(sol, layout, case, strategy, "tiled", float(col["Cp_farm"][0]), col["start"][0], float(col["seconds"][0]),
                                      col["success"][0], col["nit"][0], bool(col["bem_converged"][0]), col["code_hash"][0], col["cp_ref"][0],
                                      time.time() - t0))
    except BaseException:
        return _fail(fn)


def generate(regenerate=False):
    if regenerate and RUNDIR.exists():
        for f in RUNDIR.rglob("*.csv"):
            f.unlink()
    keys = [(case, strategy) for case, cfg in CASES.items() for strategy in cfg["strategies"]]
    t0 = time.time()
    foreach(_run, keys, context="spawn", parallel=True, processes=PROCESSES)
    foreach(_tile, [k for k in keys if _run_file(*k).exists()], context="spawn", parallel=True, processes=PROCESSES)
    print(f"optimised and tiled {len(keys)} runs in {time.time() - t0:.0f} s (PROCESSES={PROCESSES})")
    failed = sorted((RUNDIR / "failed").glob("*.txt")) if (RUNDIR / "failed").exists() else []
    if failed:
        print(f"{len(failed)} failures (tracebacks in {RUNDIR / 'failed'}): {[f.stem for f in failed]}")
    files = [_tiled_file(*k) for k in keys]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        print(f"{len(missing)} runs missing: {missing}")
    if len(files) == len(missing):
        return None
    df = pl.concat([pl.read_csv(f, infer_schema_length=None) for f in files if f.exists()], how="diagonal_relaxed")
    df.write_csv(utils.CACHEDIR / f"{FILESTEM}.csv")
    hashes = df["code_hash"].unique().to_list()
    print(f"code hashes in the output: {hashes}" + ("  <-- MIXED: sources changed during the sweep, rerun" if len(hashes) > 1 else ""))
    return df


if __name__ == "__main__":
    generate(regenerate="--regenerate" in sys.argv)
