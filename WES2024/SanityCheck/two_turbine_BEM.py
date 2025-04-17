# dualitic is imported first to ensure correct monkey patching.
import dualitic

from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from foreach import foreach
from MITRotor import IEA15MW
from mitwindfarm import BEM, Layout, Windfarm, Niayifar,VariableKwGaussianWakeModel

from WES2024 import utils
from WES2024.BEM_gradients import DualBEM
from WES2024.CustomRotors import BEMUnifiedMomentumLUT
from WES2024.optimise import (
    JointControlBEM,
    NoControlBEM,
    ThrustControlBEM,
    YawControlBEM,
    YawControlKOmegaBEM,
)

__all__ = ["generate"]

FILESTEM = Path(__file__).stem

PARALLEL = False


windfarm = Windfarm(
    rotor_model=BEM(IEA15MW(), BEM_model=DualBEM, momentum_model=BEMUnifiedMomentumLUT()),
    superposition=Niayifar(),
    wake_model=VariableKwGaussianWakeModel(0.7683081169878619, 0.0, 0.004825109405157736),
    TIamb=0.056,
)
layout = Layout([0, 6], [0.0, 0.0])
wdirs = np.arange(-20, 20, 1)
# wdirs = [0.0, 1.0, 2.0, 3.0]


methods = {
    # "YawKOmegaControl": YawControlKOmegaBEM,
    # "NoControl": NoControlBEM,
    # "YawControl": YawControlBEM,
    # "ThrustControl": ThrustControlBEM,
    "JointControl": JointControlBEM,
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

    dfs = foreach(_generate, params, context="spawn", parallel=PARALLEL)
    df = pl.concat(dfs)
    return df


if __name__ == "__main__":
    df = generate(regenerate=True)
    print(df)
