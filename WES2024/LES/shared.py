# dualitic must be imported first

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from mitwindfarm import (
    Layout,
    Square,
    GridLayout,
    WindfarmSolution,
)

# Directories
STEP_1_DIR = Path(__file__).parent / "step_1"
STEP_2_DIR = Path(__file__).parent / "step_2"
STEP_3_DIR = Path(__file__).parent / "step_3"
STEP_4_DIR = Path(__file__).parent / "step_4"
# LES_output_dir = Path(__file__).parent / "LES_output"
# LES_input_dir = Path(__file__).parent / "LES_input"

STEP_1_DIR.mkdir(exist_ok=True, parents=True)
STEP_2_DIR.mkdir(exist_ok=True, parents=True)
STEP_3_DIR.mkdir(exist_ok=True, parents=True)
STEP_4_DIR.mkdir(exist_ok=True, parents=True)


# Layouts

BASE_LAYOUT = Square(6.0, 5).rotate(45)
CALIBRATION_LAYOUT = BASE_LAYOUT.rotate(-2.5)

TIAMB = 0.056  # Determined from LES.


def normalize_by_upstream(Cp_list: list[float], row_indices: list[list[int]]) -> list[float]:
    """
    Return the normalized power output of each turbine normalized by the most
    upstream turbine. uses row indicies provided to determine upstream and
    downstream turbines.
    """
    P_norm = np.zeros_like(Cp_list)

    for row in row_indices:
        for idx in row:
            P_norm[idx] = Cp_list[idx] / Cp_list[row[0]]

    return P_norm


@dataclass
class TurbineDefinition:
    """
        Data class representing the definition of a wind turbine.
    logitech
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
    case_note: str
    turbines: list[TurbineDefinition]

    @classmethod
    def from_windfarmsolution(
        cls,
        sol: WindfarmSolution,
        casename: str,
        wdir: float,
        controller: str,
        case_note: str,
    ) -> "SimulationDefinition":
        xmin, ymin = sol.layout.x.min(), sol.layout.y.min()

        turbines = []
        for i, ((x, y, z), rotor) in enumerate(zip(sol.layout, sol.rotors)):
            _turbine = TurbineDefinition(
                f"turbine_{i}",
                x - xmin,
                y - ymin,
                z,
                np.rad2deg(rotor.yaw),
                rotor.Ctprime,
            )
            turbines.append(_turbine)

        return SimulationDefinition(
            turbines=turbines,
            casename=casename,
            wdir=wdir,
            controller=controller,
            case_note=case_note,
        )

    @classmethod
    def from_json(cls, json_fn: Path) -> "SimulationDefinition":
        with open(json_fn, "r") as f:
            data = json.load(f)

        turbines = []
        for turbine in data["turbines"]:
            turbines.append(
                TurbineDefinition(
                    turbine["turbine_ID"],
                    turbine["x"],
                    turbine["y"],
                    turbine["z"],
                    turbine["yaw"],
                    turbine["ctp"],
                )
            )

        kwargs = {
            "casename": data["casename"],
            "wdir": data["wdir"],
            "controller": data["controller"],
            "case_note": data["case_note"],
        }
        return SimulationDefinition(turbines=turbines, **kwargs)

    def write_json(self, fn) -> None:
        with open(fn, "w") as f:
            json.dump(asdict(self), f, indent=4)

    def layout(self) -> Layout:
        xs = [x.x for x in self.turbines]
        ys = [x.y for x in self.turbines]
        zs = [x.z for x in self.turbines]

        return Layout(xs, ys, zs)

    def setpoints(self) -> list[tuple[float, float]]:
        return [(turb.ctp, np.deg2rad(turb.yaw)) for turb in self.turbines]
