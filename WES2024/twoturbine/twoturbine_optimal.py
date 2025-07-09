"""
Reality check: the optimal control strategy for a two-turbine wind farm
from this script should be on the AD line at wdir 3.8 of the two-turbine
sweep if everything is consistent. 
"""
import dualitic

from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from foreach import foreach
from mitwindfarm import GridLayout, Windfarm

from WES2024 import utils
from WES2024.CustomRotors import UnifiedLUTAD
from WES2024.optimise import JointControl, NoControl, ThrustControl, YawControl
from WES2024.LES_new.final_calibration import get_wakemodel
from WES2024.LES.shared import TIAMB

__all__ = ["generate"]

FILESTEM = Path(__file__).stem
REGENERATE = True

windfarm = Windfarm(rotor_model=UnifiedLUTAD(), wake_model=get_wakemodel(), TIamb=TIAMB)
layout = GridLayout(6.0133, 0.0, 2, 1)  # unrotated 2x1 wind farm
wdirs = [3.8]  # incident wind direction

methods = {
    "NoControl": NoControl,
    "YawControl": YawControl,
    "ThrustControl": ThrustControl,
    "JointControl": JointControl,
}


def _generate(x):
    method, wdir = x
    sol = methods[method](layout.rotate(wdir), windfarm).optimise(Cp_constraint=None)

    return utils.to_polars(sol).with_columns(
        pl.lit(method).alias("method"), pl.lit(wdir).alias("wdir")
    )

@utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False):
    params = list(product(methods, wdirs))

    df = pl.concat(foreach(_generate, params, context="spawn", parallel=True))
    return df


if __name__ == "__main__":
    df = generate(regenerate=True)
    print(df.with_columns(pl.col("setpoint_0").alias("ctp_opt"), np.rad2deg(df["setpoint_1"]).alias("yaw_opt")))
