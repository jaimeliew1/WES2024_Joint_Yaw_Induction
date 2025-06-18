from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from foreach import foreach
from MITRotor.ReferenceTurbines import IEA15MW
from MITRotor.Aerodynamics import DefaultAerodynamics
from mitwindfarm import BEM, Layout, Windfarm
from rich import print
from scipy.optimize import minimize

from WES2024 import utils
from WES2024.BEM_gradients import DualBEM
from WES2024.CustomRotors import BEMUnifiedMomentumLUT

FILESTEM = Path(__file__).stem

REGENERATE = True
PARALLEL = True

PITCH_OPT = -0.023146163628916267
TSR_OPT = 9.23061314763139

bem = BEM(IEA15MW(), 
          BEM_model=DualBEM, 
          momentum_model=BEMUnifiedMomentumLUT(averaging="rotor_induction_tiploss"),
          aerodynamic_model=DefaultAerodynamics())
windfarm = Windfarm(rotor_model=bem)
layout_single = Layout(np.array([0]), np.array([0.0, 0.0]))
layout_double = Layout(np.array([0, 8]), np.array([0.0, 0.5]))

# Low res
yaws = np.arange(0, 45)
Ctprimes = np.arange(0.5, 3.01, 0.25)

# High res
yaws = np.arange(0, 45)
Ctprimes = np.arange(0.5, 3.01, 0.1)

opt_cache = {}


def get_opt_solution(yaw: float) -> tuple[float, float, float]:
    if yaw in opt_cache:
        return opt_cache[yaw]
    else:

        def to_opt(x):
            pitch, tsr = x
            return -windfarm(layout_single, [(pitch, tsr, np.deg2rad(yaw))]).Cp

        res = minimize(to_opt, (0, 9))
        pitch, tsr = res.x

        sol = windfarm(layout_single, [(pitch, tsr, np.deg2rad(yaw))]).rotors[0]
        opt_cache[yaw] = (sol.extra.pitch, sol.extra.tsr, sol.Ctprime)

        return opt_cache[yaw]


def calc_BEM_equiv_setpoint(
    bem: BEM, yaw: float, Ctprime_target: float
) -> Optional[tuple[float, float]]:
    # Get optimal set point at given yaw angle
    pitch_0, tsr_0, Ctprime_0 = get_opt_solution(yaw)

    # if Ctprime_target is too high, return None
    # if Ctprime_target > Ctprime_0 * 1.2:
    #     return None

    def to_opt(x):
        pitch, tsr = x
        return -windfarm(layout_single, [(pitch, tsr, np.deg2rad(yaw))]).Cp

    def constraint(x):
        pitch, tsr = x
        return (
            windfarm(layout_single, [(pitch, tsr, np.deg2rad(yaw))]).rotors[0].Ctprime
            - Ctprime_target
        )

    res = minimize(
        to_opt,
        (pitch_0, tsr_0),
        bounds=[(pitch_0 - np.deg2rad(5), np.deg2rad(20)), (0.5, tsr_0 * 1.2)],
        constraints=dict(type="eq", fun=constraint),
    )
    # print(res)
    if not res.success:
        return None
    sol = windfarm(
        layout_double, [(res.x[0], res.x[1], np.deg2rad(yaw)), (PITCH_OPT, TSR_OPT, 0.0)]
    )
    return sol


def _generate(x):
    Ctprime, yaw = x
    sol = calc_BEM_equiv_setpoint(bem, yaw, Ctprime)

    if sol:
        return utils.to_polars(sol).with_columns(
            pl.lit(Ctprime).alias("Ctprime_target"), pl.lit(yaw).alias("yaw_target")
        )
    else:
        return None


@utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False):
    params = list(product(Ctprimes, yaws))

    dfs = foreach(_generate, params, context="spawn", parallel=PARALLEL)
    df = pl.concat((df for df in dfs if df is not None))
    return df


if __name__ == "__main__":
    df = generate(regenerate=REGENERATE)
    print(df)
