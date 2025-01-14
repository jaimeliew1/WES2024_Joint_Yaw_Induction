from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from foreach import foreach
from MITRotor.ReferenceTurbines import IEA15MW
from mitwindfarm import BEM, Windfarm, Niayifar
from rich import print

from WES2024 import utils
from WES2024.BEM_gradients import DualBEM
from WES2024.CustomRotors import BEMUnifiedMomentumLUT
from WES2024.Generate import diamond_BEM

__all__ = ["generate"]

FILESTEM = Path(__file__).stem

PARALLEL = True
REGENERATE = True

windfarm = Windfarm(
    rotor_model=BEM(IEA15MW(), BEM_model=DualBEM, momentum_model=BEMUnifiedMomentumLUT()),
    superposition=Niayifar(),
)


def _generate(x) -> pl.DataFrame:
    _df, wdir_offset = x
    sol_0 = utils.from_polars(_df, windfarm)
    sol = windfarm(sol_0.layout.rotate(wdir_offset), sol_0.setpoints)

    return utils.to_polars(sol).with_columns(
        method=pl.lit(_df["method"][0]), wdir=_df["wdir"][0], wdir_offset=wdir_offset
    )


@utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False):
    df = diamond_BEM.generate()
    df_opt_list = [_df for _, _df in df.group_by("method", "wdir")]
    wdir_offsets = np.arange(-15.0, 15.1, 1.0)
    params = list(product(df_opt_list, wdir_offsets))

    df = pl.concat(foreach(_generate, params, context="spawn", parallel=PARALLEL))
    return df


if __name__ == "__main__":
    df = generate(regenerate=REGENERATE)
    print(df)
