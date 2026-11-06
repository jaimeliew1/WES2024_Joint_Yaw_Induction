# dualitic is imported first to ensure correct monkey patching.
import dualitic

"""
Wind direction sweep of the optimal two-turbine BEM set points (6 D aligned pair, BEM-LUT annulus-closure rotor, all six control
strategies), +-20 deg, one cache per wake model in data/generate_two_turbine_BEM_fine_<model>.csv:

    gaussBP, varvortex   0.1 deg steps, SLSQP ftol 1e-10 (noise free)
    tandem               0.2 deg steps, SLSQP ftol 1e-8, with the ground (wakemodels.get_windfarm). Marched with scipy_rk45 at rtol 1e-4 /
                         atol 1e-7: measured on the 6 D pair (fixed set points, wdir 3-4 deg in 0.02 deg steps, against an rtol 1e-8 reference)
                         rk45 at its default tolerance is no better than rk23 (bias 3e-5 in Cp), rk45 at rtol 1e-4 has a bias of 2.5e-6 and
                         wdir-to-wdir scatter of 6e-9 (rk23: 4e-6 and 1e-8), for ~1.4x the cost of a forward solve. Cost: 2.7 h on 12 workers (the K-omega^2 chains are the slowest, the other models take seconds).

Every optimised strategy is warm-started from the neighbouring wdir and chained monotonically OUTWARD from wdir = +-step (one chain per sign).
wdir = 0 is indeterminate for the yaw strategies (two symmetric optima, +-yaw) and is therefore not solved for them; it is solved (standalone)
for NoControl and ThrustControl only. The first point of each chain is a multistart (+-yaw0, best Cp kept). A Ct limit that never binds (1.5,
controllers.CT_MAX) still switches on ConstrainedMixin's scaled, ftol-controlled SLSQP path.

Usage:
    python -m WES2024.R1.generate.generate_two_turbine_BEM_fine                  # all three wake models
    python -m WES2024.R1.generate.generate_two_turbine_BEM_fine --model varvortex
    python -m WES2024.R1.generate.generate_two_turbine_BEM_fine --model tandem --wmax 2 --out test   # quick test (cached to a different file)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from foreach import foreach
from mitwindfarm import Layout

from WES2024 import utils
from WES2024.R1.controllers import METHODS, VAR_SCALES, make_controller, use_annulus_optimum
from WES2024.R1.wakemodels import get_bem_lut_rotor, get_windfarm

use_annulus_optimum()

__all__ = ["generate", "load", "MODELS"]

FILESTEM = Path(__file__).stem

layout = Layout([0, 6], [0.0, 0.0])
WDIR_MAX = 20.0
INTEGRATOR = "scipy_rk45"
RTOL, ATOL = 1e-4, 1e-7
MAXITER = 200
PROCESSES = 12  # one worker per chain (5 strategies x 2 signs) + the 2 standalone tasks

MODELS = ["gaussBP", "varvortex", "tandem"]
# wind direction step [deg] and SLSQP ftol per wake model. Only TANDEM marches a PDE (integrator noise, grid floor), so only it needs the looser ftol.
MODEL_CFG = {
    "tandem": dict(step=0.2, ftol=1e-8),
    "gaussBP": dict(step=0.1, ftol=1e-10),
    "varvortex": dict(step=0.1, ftol=1e-10),
}
# multistart yaw guesses [rad] for the first point of a chain (the yaw strategies have ~symmetric +-yaw optima there);
# +-0.3 rad starts were checked to land on the same two optima
STARTS = {m: (0.05, -0.05) for m in VAR_SCALES}
STARTS["ThrustControl"] = (0.05,)  # no yaw variable


def _windfarm(model="tandem"):
    # the ivp settings only mean something to the marched (curled) wake model
    overrides = dict(integrator=INTEGRATOR, ivp_kwargs=dict(rtol=RTOL, atol=ATOL)) if model == "tandem" else None
    return get_windfarm(model, rotor_model=get_bem_lut_rotor(), solver_overrides=overrides)


def _row(sol, method, wdir, model="tandem"):
    return utils.to_polars(sol).with_columns(
        pl.lit(model).alias("model"), pl.lit(method).alias("method"), pl.lit(float(wdir)).alias("wdir")
    )


def _solve(method, wdir, windfarm, x0=None, yaw0=0.05, model="tandem"):
    """One SLSQP solve; warm-started at x0 if given. Returns (solution, optimum x, nit, wall seconds)."""
    controller = make_controller(method, layout.rotate(wdir), windfarm, model, MODEL_CFG[model]["ftol"], MAXITER, yaw0=yaw0, x0=x0)
    t0 = time.time()
    sol = controller.optimise(Cp_constraint=None, use_gradients=True)
    return sol, controller.result.x.copy(), int(controller.result.nit), time.time() - t0


def _multistart(method, wdir, windfarm, model="tandem"):
    best = None
    for yaw0 in STARTS[method]:
        sol, x, nit, dt = _solve(method, wdir, windfarm, yaw0=yaw0, model=model)
        if best is None or float(sol.Cp) > best[0]:
            best = (float(sol.Cp), sol, x)
    return best[1], best[2]


def _chain(task):
    """Optimise `method` at wdirs (ordered outward from +-step): multistart at the first point, then each point
    warm-started from the previous point's optimum. Returns per-point rows and (nit, seconds) diagnostics."""
    model, method, wdirs = task
    windfarm = _windfarm(model)
    sol, x0 = _multistart(method, wdirs[0], windfarm, model)
    rows, diag = [_row(sol, method, wdirs[0], model)], []
    for wdir in wdirs[1:]:
        sol, x0, nit, dt = _solve(method, wdir, windfarm, x0=x0, model=model)
        rows.append(_row(sol, method, wdir, model))
        diag.append((method, wdir, nit, dt))
    return rows, diag


def _standalone(task):
    """Independent solves for the strategies with no chain: NoControl (no variables) at every wdir, and the wdir=0
    point of ThrustControl (symmetric in wdir, no yaw tie to break)."""
    model, method, wdirs = task
    windfarm = _windfarm(model)
    rows = []
    for wdir in wdirs:
        if method == "NoControl":
            sol = METHODS[method](layout.rotate(wdir), windfarm).optimise(Cp_constraint=None)
        else:
            sol, _ = _multistart(method, wdir, windfarm, model)
        rows.append(_row(sol, method, wdir, model))
    return rows, []


def _chain_or_standalone(task):
    model, method, wdirs = task
    if method == "NoControl" or (method == "ThrustControl" and wdirs == [0.0]):
        return _standalone(task)
    return _chain(task)


def _cache(model, out=None):
    return utils.CACHEDIR / f"{out or f'{FILESTEM}_{model}'}.csv"


def generate(model="tandem", step=None, wmax=None, out=None, regenerate=False, processes=PROCESSES):
    fn = _cache(model, out)
    if fn.exists() and not regenerate:
        return pl.read_csv(fn)

    step, wmax = step or MODEL_CFG[model]["step"], wmax or WDIR_MAX
    pos = np.round(np.arange(1, int(round(wmax / step)) + 1) * step, 6)  # +step ... +wmax (no wdir=0)
    tasks = [(model, m, list(sign * pos)) for m in VAR_SCALES for sign in (1.0, -1.0)]
    tasks += [(model, "NoControl", list(np.concatenate([-pos[::-1], [0.0], pos])))]
    tasks += [(model, "ThrustControl", [0.0])]  # wdir=0 for ThrustControl (NoControl's list above already has it)
    t0 = time.time()
    results = foreach(_chain_or_standalone, tasks, context="spawn", parallel=True, processes=processes)
    rows = [r for rs, _ in results for r in rs]
    diag = [d for _, ds in results for d in ds]
    print(f"{model}: sweep wall time {time.time() - t0:.0f} s")
    if diag:
        d = pl.DataFrame(diag, schema=["method", "wdir", "nit", "seconds"], orient="row")
        print(d.group_by("method").agg(pl.col("nit").mean(), pl.col("nit").max().alias("nit_max"),
                                       pl.col("seconds").mean(), pl.col("seconds").sum().alias("total_s")))
    df = pl.concat(rows, how="diagonal_relaxed").sort("method", "wdir", "turbine")
    fn.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(fn)
    return df


def load(out=None, model="tandem"):
    return pl.read_csv(_cache(model, out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=list(MODEL_CFG), help="default: all three wake models")
    ap.add_argument("--step", type=float, default=None, help="wind direction step [deg] (default: MODEL_CFG)")
    ap.add_argument("--wmax", type=float, default=WDIR_MAX)
    ap.add_argument("--out", default=None, help="cache file stem in data/ (default: this script's stem + _model)")
    ap.add_argument("--processes", type=int, default=PROCESSES)
    args = ap.parse_args()
    for m in [args.model] if args.model else MODELS:
        print(generate(m, args.step, args.wmax, args.out, regenerate=True, processes=args.processes))
