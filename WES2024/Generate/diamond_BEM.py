from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from foreach import foreach
from MITRotor.ReferenceTurbines import IEA15MW
from mitwindfarm.Layout import Square
from mitwindfarm.Rotor import BEM
from mitwindfarm.windfarm import Windfarm
from WES2024.BEM_gradients import DualBEM

from WES2024 import utils
from WES2024.optimise import JointControlBEM, NoControlBEM, ThrustControlBEM, YawControlBEM

__all__ = ["generate"]

FILESTEM = Path(__file__).stem


PARALLEL = True


windfarm = Windfarm(rotor_model=BEM(IEA15MW(), BEM_model=DualBEM))
layout = Square(6.0, 5).rotate(45)
wdirs = np.arange(0.0, 90.0, 0.05)


methods = {
    "NoControl": NoControlBEM,
    "YawControl": YawControlBEM,
    "ThrustControl": ThrustControlBEM,
    "JointControl": JointControlBEM,
}


def _generate(x):
    method, wdir = x
    sol = methods[method](layout.rotate(wdir), windfarm).optimise(
        Cp_constraint=None,
        use_gradients=True,
        verbose=False,
    )

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
