from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from foreach import foreach
from mitwindfarm.Layout import Layout
from mitwindfarm.windfarm import Windfarm

from WES2024 import utils
from WES2024.optimise import JointControl, NoControl, ThrustControl, YawControl

__all__ = ["generate"]

FILESTEM = Path(__file__).stem


windfarm = Windfarm()
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

    df = pl.concat(foreach(_generate, params, parallel=True))
    return df


if __name__ == "__main__":
    df = generate()
    print(df)
