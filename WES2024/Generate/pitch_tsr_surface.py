import itertools
from pathlib import Path

import numpy as np
import polars as pl
import foreach
from MITRotor.ReferenceTurbines import IEA15MW
from mitwindfarm.Layout import Layout
from mitwindfarm.Rotor import BEM
from mitwindfarm.windfarm import Windfarm

from WES2024 import utils

__all__ = ["generate"]

FILESTEM = Path(__file__).stem

layout = Layout(np.array([0.0]), np.array([0.0]))
pitches = np.linspace(-15, 15, 150)
tsrs = np.linspace(5, 15, 150)
yaws = np.arange(0.0, 50.1, 5.0)

windfarm = Windfarm(rotor_model=BEM(IEA15MW()))


def _generate(x) -> pl.DataFrame:
    pitch, tsr, yaw = x
    setpoints = [(np.deg2rad(pitch), tsr, np.deg2rad(yaw))]
    sol = windfarm(layout, setpoints)
    df = utils.to_polars(sol)
    return df


@utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False) -> pl.DataFrame:
    params = list(itertools.product(pitches, tsrs, yaws))

    df_list = foreach(_generate, params, parallel=True)

    return pl.concat(df_list)


if __name__ == "__main__":
    df = generate(regenerate=True)
    print(df)
