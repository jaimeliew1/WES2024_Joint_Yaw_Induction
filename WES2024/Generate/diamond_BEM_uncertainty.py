from itertools import product
from pathlib import Path

import polars as pl
import numpy as np
import foreach
from mitwindfarm.windfarm import Windfarm
from MITRotor.ReferenceTurbines import IEA15MW
from mitwindfarm.Rotor import BEM

from WES2024 import utils
from WES2024.Generate import diamond_BEM
from WES2024.BEM_gradients import DualBEM


from rich import print

__all__ = ["generate"]

FILESTEM = Path(__file__).stem


windfarm = Windfarm(rotor_model=BEM(IEA15MW(), BEM_model=DualBEM))


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
    wdir_offsets = np.arange(-10.0, 10.0, 1.0)
    params = list(product(df_opt_list, wdir_offsets))

    df = pl.concat(foreach(_generate, params, context="spawn", parallel=True))
    return df


if __name__ == "__main__":
    df = generate()
    print(df)
