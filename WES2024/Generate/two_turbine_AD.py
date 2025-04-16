# dualitic is imported first to ensure correct monkey patching.
import dualitic

from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from foreach import foreach
from mitwindfarm import Layout, Windfarm, Niayifar, VariableKwGaussianWakeModel

from WES2024 import utils
from WES2024.CustomRotors import UnifiedLUTAD
from WES2024.optimise import JointControl, NoControl, ThrustControl, YawControl
from WES2024.LES_new.final_calibration import get_wakemodel
from WES2024.LES.shared import TIAMB

__all__ = ["generate"]

FILESTEM = Path(__file__).stem
REGENERATE = True

windfarm = Windfarm(
    rotor_model=UnifiedLUTAD(), 
    wake_model=get_wakemodel(), 
    TIamb=TIAMB,
    superposition=Niayifar(),
)
layout = Layout([0, 6.01], [0.0, 0.0])
wdirs = np.arange(-20, 20, 0.05)


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
    df = generate(regenerate=REGENERATE)
    print(df)
