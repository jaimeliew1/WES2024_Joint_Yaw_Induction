# dualitic is imported first to ensure correct monkey patching.
import dualitic

"""
R1: two-turbine test following the setup of Tamaro et al. (WES 2024): two IEA 15 MW BEM rotors at 5 D spacing, 6 %
ambient turbulence intensity, wind direction sweep, with the TANDEM, varvortex (Vortex), BP2016 Gaussian and original axisymmetric Gaussian ("gauss") wake models. Greedy
(NoControl), yaw with the K-omega^2 torque law (pitch fixed) and joint (pitch, TSR, yaw) control; same controllers and
bounds as generate_two_turbine_BEM_fine.py. Results are cached per wake model in
data/generate_two_turbine_tamaro_{model}.csv (turbine 0 is the leading turbine). The default is the two Gaussian models of the referee
response figure (forpaper/plot_two_turbine_tamaro_vs_paper.py).

The rotor is the tabulated uniform-inflow BEM (generate_bem_lut_annulus.py), ~5x cheaper per TANDEM optimisation than the BEM rotor.

Optimiser (R1.controllers.ConstrainedMixin): SLSQP with AD gradients on scaled variables (pitch 0.05 rad, TSR 1,
yaw 0.3 rad; Ct_max is set for all models, it only binds for BP2016). optimise.py's default unscaled L-BFGS-B stops early on
the flat pitch/TSR ridge. The ridge is very flat: the farm Cp changes by only ~5e-4 over 0.5 in TSR, so the stopping
tolerance and the noise of Cp decide where on the ridge an optimisation ends.
  - varvortex, gaussBP, gauss: Cp is noise free, ftol = 1e-11.
  - tandem: with the default adaptive-integrator tolerance (rtol 1e-3) Cp carries ~1e-6 noise, so SLSQP (ftol 1e-6) stops
    wherever the noise says and TSR/pitch scatter from one wind direction to the next (and stay near the initial guess
    TSR_OPT). Stage 1 (rtol 1e-3, ftol 1e-6) finds the basin; stage 2 restarts from it with rtol 1e-5 (Cp noise ~1e-8)
    and ftol 1e-10, which converges to a unique, smooth optimum (two yaw starts agree to ~1e-8 in Cp). ~10x costlier.
Joint and yaw/K-omega control are started from two yaw guesses and the better optimum is kept: the yaw cap (30 deg) binds
for |wdir| <~ 4 deg and, e.g., varvortex has two local optima at wdir = 0 (Cp differing by 5e-5).

Usage: python -m WES2024.R1.generate.generate_two_turbine_tamaro [model ...]
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from foreach import foreach
from mitwindfarm import Area, Layout

from WES2024 import utils
from WES2024.R1.controllers import METHODS, make_controller, use_annulus_optimum
from WES2024.R1.wakemodels import get_bem_lut_rotor, get_windfarm

__all__ = ["generate", "methods", "MODELS", "SPACING", "TIAMB", "FILESTEM"]

use_annulus_optimum()

FILESTEM = Path(__file__).stem
MODELS = ["gauss", "gaussBP"]  # also runs "varvortex" and "tandem"
SPACING = 5.0  # [D]
TIAMB = 0.06
layout = Layout([0, SPACING], [0.0, 0.0])
wdirs = np.arange(-20, 20.01, 1.0)

YAW0 = {"tandem": 5e-2}  # initial yaw guess [rad] per wake model
YAW0_DEFAULT = 1e-2
YAW0_LARGE = 0.3  # second start, to find the best part of the ridge
FTOL = {"tandem": 1e-6, "varvortex": 1e-11, "gaussBP": 1e-11, "gauss": 1e-11}
MAXITER = {"tandem": 100, "varvortex": 400, "gaussBP": 400, "gauss": 400}
TANDEM_POLISH = dict(ftol=1e-10, maxiter=300, ivp_kwargs=dict(rtol=1e-5, atol=1e-8))  # stage 2 for TANDEM

methods = {k: METHODS[k] for k in ("NoControl", "YawKOmegaControl", "JointControl")}


def _windfarm(model, polish=False):
    overrides = dict(ivp_kwargs=TANDEM_POLISH["ivp_kwargs"]) if polish else None
    return get_windfarm(model, TIamb=TIAMB, rotor_model=get_bem_lut_rotor(rotor_grid=Area()), solver_overrides=overrides)


def _solve(model, method, wdir, yaw0, x0=None, polish=False):
    """One optimisation; returns (solution, optimiser result). x0 (variable vector) overrides the yaw start."""
    ftol = TANDEM_POLISH["ftol"] if polish else FTOL[model]
    maxiter = TANDEM_POLISH["maxiter"] if polish else MAXITER[model]
    controller = make_controller(method, layout.rotate(wdir), _windfarm(model, polish), model, ftol, maxiter, yaw0=yaw0, x0=x0)
    return controller.optimise(Cp_constraint=None, use_gradients=True), controller.result


def _generate(x):
    model, method, wdir = x
    if method == "NoControl":
        sol = methods[method](layout.rotate(wdir), _windfarm(model)).optimise()
    else:
        yaw0 = YAW0.get(model, YAW0_DEFAULT)
        runs = [_solve(model, method, wdir, y) for y in (yaw0, YAW0_LARGE)]
        sol, res = max(runs, key=lambda r: float(r[0].Cp))
        if model == "tandem":
            sol, res = _solve(model, method, wdir, yaw0, x0=res.x, polish=True)
    return utils.to_polars(sol).with_columns(
        pl.lit(model).alias("model"), pl.lit(method).alias("method"), pl.lit(wdir).alias("wdir")
    )


def generate(model, regenerate=False):
    fn = utils.CACHEDIR / f"{FILESTEM}_{model}.csv"
    if fn.exists() and not regenerate:
        return pl.read_csv(fn)
    params = [(model, m, w) for m, w in product(methods, wdirs)]
    df = pl.concat(foreach(_generate, params, context="spawn", parallel=True, processes=36))
    df.write_csv(fn)
    return df


if __name__ == "__main__":
    for model in sys.argv[1:] or MODELS:
        print(model)
        print(generate(model, regenerate=True))
