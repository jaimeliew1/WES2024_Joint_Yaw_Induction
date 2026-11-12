# dualitic is imported first to ensure correct monkey patching.
import dualitic

"""
R1: forward sweeps of the leading turbine's (Ct', yaw) of the two-turbine LES (6.0133 D spacing, waked turbine at Ct' = 2, yaw = 0) for the contour
figure forpaper/twoturbine_gain_contours.py. The rotor models are actuator disks driven by (Ct', yaw), as the LES:

    cosine_vargauss    CosineAD, P = P(gamma = 0) cos^Pp(gamma), Pp = 1.98 calibrated to the LES-based Ct' = 2 power-yaw relation,
                       with the Gaussian wake kw = a TI + b, a = 0.636, b = 0
    unified_vargauss   Unified Momentum Model rotor (UnifiedAD_veer) with the same Gaussian wake
    unified_vortex     UMM rotor with the vortex sheet (Niayifar kw, z_wall = -1; no ustar, unlike wakemodels.get_windfarm("varvortex"))
    unified_tandem     UMM rotor with TANDEM (wakemodels.get_windfarm("tandem"), with the ground)

Two grids per configuration, cached in data/generate_two_turbine_contours_<key>[_dense].csv with columns ctp, yaw [deg], Cp_1, Cp_2, Cp:
    sparse   each LES run's exact (Ct', yaw, hub wind direction phi_hub), so that it lines up point-by-point with the LES (the MAE)
    dense    regular 21 x 19 grid (twice the LES resolution) at the fixed wind direction 3.8 deg, for the contours and the model optima

Usage: python -m WES2024.R1.generate.generate_two_turbine_contours   (~10 min, one worker per sweep)
"""

from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from foreach import foreach
from mitwindfarm import Area, GridLayout, VariableKwGaussianWakeModel, VariableVortexWakeModel, Windfarm
from mitwindfarm.Rotor import UnifiedAD_veer

from WES2024 import utils
from WES2024.CustomRotors import CosineAD
from WES2024.R1 import wakemodels
from WES2024.R1.les_data import TWOTURBINE_LES
from WES2024.R1.wakemodels import TIAMB

FILESTEM = Path(__file__).stem
KEYS = ["cosine_vargauss", "unified_vargauss", "unified_vortex", "unified_tandem"]
PP_CALIBRATED = 1.98
SPACING = 6.0133  # [D]
WDIR_LES = 3.81  # [deg] wind direction of the sweep, before the correction by each run's phi_hub
WDIR_DENSE = 3.8  # [deg]
SP2 = (2, 0)  # the waked turbine is always at Betz
N_CTP_DENSE, N_YAW_DENSE = 21, 19


def get_windfarm(key):
    rotor = CosineAD(rotor_grid=Area(), Pp=PP_CALIBRATED) if key == "cosine_vargauss" else UnifiedAD_veer(rotor_grid=Area())
    if key == "unified_tandem":
        return wakemodels.get_windfarm("tandem", TIamb=TIAMB, rotor_model=rotor)
    wake = VariableVortexWakeModel(z_wall=-1) if key == "unified_vortex" else VariableKwGaussianWakeModel(a=0.636, b=0, c=0)
    return Windfarm(rotor_model=rotor, wake_model=wake, TIamb=TIAMB)


def _to_df(solutions):
    return pl.DataFrame({
        "ctp": [sol.rotors[0].Ctprime for sol in solutions],
        "yaw": np.rad2deg([sol.rotors[0].yaw for sol in solutions]),
        "Cp_1": [sol.rotors[0].Cp for sol in solutions],
        "Cp_2": [sol.rotors[1].Cp for sol in solutions],
        "Cp": [sol.Cp for sol in solutions],
    })


def _run_sparse(key):
    wf = get_windfarm(key)
    layout = GridLayout(SPACING, 0.0, 2, 1).rotate(WDIR_LES)
    les = pl.read_csv(TWOTURBINE_LES).select("yaw", "ctp", "phi_hub")
    return _to_df([wf(layout.rotate(-np.degrees(phi)), [(ctp, np.deg2rad(yaw)), SP2]) for yaw, ctp, phi in les.iter_rows()])


def _run_dense(key):
    wf = get_windfarm(key)
    layout = GridLayout(SPACING, 0.0, 2, 1).rotate(WDIR_DENSE)
    ctps = np.linspace(0.4, 4.4, N_CTP_DENSE)
    yaws = np.deg2rad(np.linspace(0, 45, N_YAW_DENSE))
    return _to_df([wf(layout, [sp1, SP2]) for sp1 in product(ctps, yaws)])


def cache_path(key, dense=False):
    return utils.CACHEDIR / f"{FILESTEM}_{key}{'_dense' if dense else ''}.csv"


def generate(key, dense=False, regenerate=False):
    fn = cache_path(key, dense)
    if fn.exists() and not regenerate:
        return pl.read_csv(fn)
    df = (_run_dense if dense else _run_sparse)(key)
    fn.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(fn)
    return df


def _task(x):
    generate(*x, regenerate=True)
    return cache_path(*x).name


if __name__ == "__main__":
    tasks = [(key, dense) for key in KEYS for dense in (False, True)]
    print(foreach(_task, tasks, context="spawn", parallel=True, processes=len(tasks)))
