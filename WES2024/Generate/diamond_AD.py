from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from foreach import foreach
from mitwindfarm import Square, Windfarm

from WES2024 import utils
from WES2024.CustomRotors import UnifiedLUTAD
from WES2024.optimise import JointControl, NoControl, ThrustControl, YawControl

# __all__ = ["generate"]

FILESTEM = Path(__file__).stem

# windfarm = Windfarm()
windfarm = Windfarm(rotor_model=UnifiedLUTAD())


methods = {
    "NoControl": NoControl,
    "YawControl": YawControl,
    "ThrustControl": ThrustControl,
    "JointControl": JointControl,
}


layouts = {
    4: Square(4.0, 5).rotate(45),
    5: Square(5.0, 5).rotate(45),
    6: Square(6.0, 5).rotate(45),
    7: Square(7.0, 5).rotate(45),
    8: Square(8.0, 5).rotate(45),
    9: Square(9.0, 5).rotate(45),
    10: Square(10.0, 5).rotate(45),
}
wdirs = np.arange(0.0, 90.0, 0.05)


def _generate(x):
    method, wdir, min_dist = x
    sol = methods[method](layouts[min_dist].rotate(wdir), windfarm).optimise(use_gradients=True)
    return utils.to_polars(sol).with_columns(
        pl.lit(method).alias("method"),
        pl.lit(wdir).alias("wdir"),
        pl.lit(min_dist).alias("min_dist"),
    )


@utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False):
    params = list(product(methods, wdirs, layouts))

    df = pl.concat(foreach(_generate, params, context="spawn", parallel=True))
    return df


if __name__ == "__main__":
    df = generate(regenerate=True)
    print(df)
