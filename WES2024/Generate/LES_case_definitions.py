import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from foreach import foreach
from mitwindfarm.Layout import Square
from mitwindfarm.windfarm import Windfarm

from WES2024 import utils
from WES2024.optimise import JointControl, NoControl, ThrustControl, YawControl

__all__ = [
    "base_layout",
    "generate",
]


REGENERATE = False

OUTPUT_DIR = Path(__file__).parent.parent.parent / "LES_cases"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

FILESTEM = Path(__file__).stem


base_layout = Square(6.0, 5).rotate(45)
windfarm = Windfarm()


wdirs = [0.0, -2.5, 2.5, 45.0, 42.0, 48.0]

controllers = {
    "nocontrol": NoControl,
    "jointcontrol": JointControl,
    "yawcontrol": YawControl,
    "thrustcontrol": ThrustControl,
}


@dataclass
class TurbineDefinition:
    """
    Data class representing the definition of a wind turbine.

    Attributes:
    - x (float): X-coordinate of the turbine in rotor diamters.
    - y (float): Y-coordinate of the turbine in rotor diamters.
    - z (float): Z-coordinate (height) of the turbine in rotor diamters.
    - yaw (float): Yaw angle of the turbine in degrees (positive is anti-clockwise).
    - ctp (float): Local thrust coefficient of the turbine.
    """

    turbine_ID: str
    x: float
    y: float
    z: float
    yaw: float
    ctp: float


@dataclass
class SimulationDefinition:
    """
    Data class representing the definition of a wind farm simulation.

    Attributes:
    - casename (str): Name of the simulation case.
    - wdir (float): Wind direction for the simulation in degrees.
    - turbines (List[TurbineDefinition]): List of TurbineDefinition objects
      representing the turbines in the wind farm.
    """

    casename: str
    wdir: float
    controller: str
    turbines: list[TurbineDefinition]


def _generate(x):
    controller, wdir = x
    sol = controllers[controller](base_layout.rotate(wdir), windfarm).optimise(use_gradients=True)
    return utils.to_polars(sol).with_columns(
        pl.lit(controller).alias("controller"),
        pl.lit(wdir).alias("wdir"),
    )


@utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False):
    params = list(product(controllers.keys(), wdirs))

    df = pl.concat(foreach(_generate, params, parallel=True))
    return df


def make_sim_cases(df: pl.DataFrame) -> SimulationDefinition:
    out = []
    for (wdir, controller), _df in df.group_by("wdir", "controller"):
        casename = f"diamond_wdir{wdir}_{controller}"
        turbines = [
            TurbineDefinition(f"turbine_{i}", *x)
            for i, x in enumerate(
                _df.select(
                    pl.col("x") - pl.col("x").min(),
                    pl.col("y") - pl.col("y").min(),
                    "z",
                    np.rad2deg(pl.col("yaw")).round(6),
                    "setpoint_0",
                ).rows()
            )
        ]
        out.append(SimulationDefinition(casename, wdir, controller, turbines))
    return out


def main():
    df = generate(regenerate=REGENERATE)
    simulations = make_sim_cases(df)

    for simulation in simulations:
        sim_dict = asdict(simulation)
        with open(OUTPUT_DIR / f"{simulation.casename}.json", "w") as f:
            json.dump(sim_dict, f, indent=4)

    # Overview dataframe
    df_overview = pl.from_dicts(
        [dict(casename=x.casename, wdir=x.wdir, controller=x.controller) for x in simulations]
    ).sort("wdir")
    df_overview.write_csv(OUTPUT_DIR / "overview.csv")


if __name__ == "__main__":
    main()
