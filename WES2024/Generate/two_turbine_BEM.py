from itertools import product
from pathlib import Path
from functools import partial

import numpy as np
import polars as pl
import foreach
from MITRotor.ReferenceTurbines import IEA15MW
from mitwindfarm.Layout import Layout
from mitwindfarm.Rotor import BEM
from mitwindfarm.windfarm import Windfarm
from WES2024.BEM_gradients import DualBEM

from WES2024 import utils
from WES2024.optimise import (
    JointControlBEM,
    NoControlBEM,
    ThrustControlBEM,
    YawControlBEM,
    YawControlKOmegaBEM,
)

__all__ = ["generate"]

FILESTEM = Path(__file__).stem

PARALLEL = True


windfarm = Windfarm(rotor_model=BEM(IEA15MW(), BEM_model=DualBEM))
layout = Layout([0, 6], [0.0, 0.0])
wdirs = np.arange(-20, 20, 0.05)
# wdirs = [0.0, 1.0, 2.0, 3.0]


methods = {
    "YawKOmegaControl": YawControlKOmegaBEM,
    "NoControl": NoControlBEM,
    "YawControl": YawControlBEM,
    "ThrustControl": ThrustControlBEM,
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

    dfs = foreach(_generate, params, parallel=PARALLEL)
    df = pl.concat(dfs)
    return df


if __name__ == "__main__":
    df = generate()
    print(df)
