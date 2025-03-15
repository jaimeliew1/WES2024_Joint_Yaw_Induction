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

__all__ = ["generate"]

FILESTEM = Path(__file__).stem
REGENERATE = True

windfarm = Windfarm(
    rotor_model=UnifiedLUTAD(),
    superposition=Niayifar(),
    wake_model=VariableKwGaussianWakeModel(0.7683081169878619, 0.0, 0.004825109405157736, x0=1.0),
    TIamb=0.056,
)
layout = Layout([0, 5], [0.0, 0.0])
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
