# dualitic must be imported first
from dualitic import DualVariables

import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mitwindfarm import (
    Layout,
    Square,
    VariableKwGaussianWakeModel,
    Windfarm,
    WindfarmSolution,
)
from scipy.optimize import minimize
from rich import print

from WES2024 import utils
from WES2024.optimise import JointControl, NoControl, ThrustControl, YawControl
from WES2024.CustomRotors import UnifiedLUTAD

LES_output_dir = Path(__file__).parent / "LES_output"
LES_input_dir = Path(__file__).parent / "LES_input"

assert LES_output_dir.exists()
LES_input_dir.mkdir(exist_ok=True, parents=True)

CASE_NOTE = "test_1.0"

LES_FN_REGEX = re.compile("(\w+)_wdir(-?\d+.\d+)_(\w+).csv")

BASE_LAYOUT = Square(6.0, 5).rotate(45)

TIAMB = 0.053  # Determined from LES.

CONTROLLERS = {
    "nocontrol": NoControl,
    "yawcontrol": YawControl,
    "thrustcontrol": ThrustControl,
    "jointcontrol": JointControl,
}


_DIAMOND_ROW_INDICES = [
    [24],
    [23, 19],
    [22, 18, 14],
    [21, 17, 13, 9],
    [20, 16, 12, 8, 4],
    [15, 11, 7, 3],
    [10, 6, 2],
    [5, 1],
    [0],
]

_SQUARE_ROW_INDICES = None

ROW_INDICES = {
    -2.5: _DIAMOND_ROW_INDICES,
    0.0: _DIAMOND_ROW_INDICES,
    42.0: _SQUARE_ROW_INDICES,
    45.0: _SQUARE_ROW_INDICES,
}


def extract_fn_params(fn: str) -> dict:
    """
    Extracts simulation type, wind direction and control method from filename of
    LES output data.
    """
    groups = LES_FN_REGEX.match(fn)
    if groups is None:
        raise ValueError("Could not match filename regex.")
    groups = groups.groups()
    out = dict(sim_method=groups[0], wdir=float(groups[1]), method=groups[2])
    return out


def retrieve_LES_output_files(
    LES_dir: Path, calibration_dir: Path | None = None, overwrite=True
) -> list[Path]:
    """
    Returns a list of LES simulation data to be used in callibration. If
    overwrite=True, ignores already callibrated cases.
    """

    files = [fn for fn in LES_dir.glob("*_nocontrol.csv")]

    if not overwrite:
        to_keep = []
        for file in files:
            params = extract_fn_params(file.name)
            if len(list(calibration_dir.glob(f"diamond_wdir{params['wdir']}*"))) == 0:
                to_keep.append(file)
        files = to_keep

    return files


def normalize_by_upstream(
    Cp_list: list[float], row_indices: list[list[int]] = ROW_INDICES
) -> list[float]:
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


class MultiSimCalibrationCase:
    def __init__(
        self,
        wdir: float,
        TIamb: float,
        row_indices: list[list[int]],
        base_layout: Layout = BASE_LAYOUT,
    ):
        self.wdir = wdir
        self.TIamb = TIamb
        self.row_indices = row_indices

        self._base_layout = base_layout
        self.layout = base_layout.rotate(wdir)

        self.calibrated = False
        self._calibration_setpoints = None

    def run_model(
        self, a: float, b: float, c: float, setpoints: list[tuple[float, float]]
    ) -> WindfarmSolution:
        """
        Run the wind farm model using a variable-wake spreading rate wake model
        for a given set of wake model parameters, (a, b, c).
        """
        wakemodel = VariableKwGaussianWakeModel(a, b, c)
        windfarm = Windfarm(
            rotor_model=UnifiedLUTAD(), wake_model=wakemodel, TIamb=self.TIamb
        )

        # setpoints = len(self.layout) * [(2.0, 0.0)]  # should this be 2.1..?
        sol = windfarm(self.layout, setpoints)

        return sol

    def set_calibration(self, a: float, b: float, c: float):
        self._calibration_setpoints = (a, b, c)
        self.calibrated = True

    def calibrate(
        self,
        Cp_ref_sets: list[list[float]],
        control_setpoint_sets: list[list[tuple[float, float]]],
        x0: tuple[float, ...] = (1.0, 1.0, 0.0),
    ) -> "MultiSimCalibrationCase":
        """
        Calibrate a polynomial mapping between TI at the rotor and wake spreading
        wake by minimizing the square error of Cp NORMALIZED by the upstream turbines.
        """
        assert len(Cp_ref_sets) == len(control_setpoint_sets)
        p_norm_refs = [
            normalize_by_upstream(Cp_ref, row_indices=self.row_indices)
            for Cp_ref in Cp_ref_sets
        ]

        def func(x):
            a, b, c = x

            cost = 0.0
            for control_setpoints, p_norm_ref in zip(
                control_setpoint_sets, p_norm_refs
            ):
                sol = self.run_model(a, b, c, control_setpoints)
                Cp_model = np.array([x.Cp for x in sol.rotors])
                p_norm = normalize_by_upstream(Cp_model, self.row_indices)

                cost += np.sum((p_norm - p_norm_ref) ** 2)

            return cost

        opt_sol = minimize(
            func,
            x0,
            bounds=[(0, 5), (-5, 5), (-1, 1)],
        )

        print(opt_sol)

        self._calibration_setpoints = opt_sol.x
        self.calibrated = True

        return self

    def setpoints(self):
        if not self.calibrated:
            raise ValueError("Case not yet calibrated.")
        else:
            return self._calibration_setpoints

    def calibrated_windfarm(self) -> Windfarm:
        a, b, c = self.setpoints()
        wakemodel = VariableKwGaussianWakeModel(a, b, c)
        windfarm = Windfarm(
            rotor_model=UnifiedLUTAD(), wake_model=wakemodel, TIamb=self.TIamb
        )

        return windfarm


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


def make_sim_case(sol: WindfarmSolution, wdir: float, controller: str) -> dict:

    xmin, ymin = sol.layout.x.min(), sol.layout.y.min()
    casename = f"diamond_wdir{wdir}_{controller}"

    turbines = []
    for i, ((x, y, z), rotor) in enumerate(zip(sol.layout, sol.rotors)):
        _turbine = TurbineDefinition(
            f"turbine_{i}", x - xmin, y - ymin, z, np.rad2deg(rotor.yaw), rotor.Ctprime
        )
        turbines.append(_turbine)

    return asdict(SimulationDefinition(casename, wdir, controller, CASE_NOTE, turbines))


def plot_text_on_layout(layout: Layout, vals: list, fn: Path, title=None):
    plt.figure()
    plt.axis("equal")

    cmap = plt.cm.viridis
    norm = colors.Normalize(vmin=np.min(vals), vmax=np.max(vals))
    for idx, (x, y, val) in enumerate(zip(layout.x, layout.y, vals)):
        plt.plot(x, y, ".", ms=10, c=cmap(norm(val)))
        plt.text(x, y, f"{val:2.3f}")

    if title:
        plt.title(title)

    plt.savefig(fn, dpi=500, bbox_inches="tight")
    plt.close()


def calibrate_wake_model(
    wdir: float,
    LES_fns: list[Path | str],
    input_fns: list[Path | str],
    calibration_fn: Optional[Path | str],
) -> None:
    """
    Calibrate wake spreading rate as a function of added wake turbulence based
    on LES results (LES_fn). Saves results as a text file (calibration_fn).

    inputs: LES results filepath (LES_fn)
    outputs: Wake calibration file (calibration_fn)
    """
    powers = [pl.read_csv(fn)["Cp"].to_numpy() for fn in LES_fns]
    setpoints = [extract_setpoints_from_json(fn) for fn in input_fns]

    calibration = MultiSimCalibrationCase(wdir, TIAMB, ROW_INDICES[wdir])
    calibration = calibration.calibrate(powers, setpoints)

    output = ",".join(str(x) for x in calibration.setpoints())

    # Write calibration values to file.

    Path(calibration_fn).parent.mkdir(exist_ok=True, parents=True)
    with open(calibration_fn, "w") as f:
        f.write(output)

    return None


def no_control_setpoints(wdir: float, setpoint_fn: Path | str) -> None:
    """
    Saves the turbine locations and set points to a JSON file (setpoint_fn) for
    use in an LES simulation.

    inputs: wind direction (wdir)
    outputs: setpoint JSON file (setpoint_fn)
    """
    setpoints = 25 * [(2.1047, 0.0)]  # should this be 2.1.. yes probably?

    calibration = MultiSimCalibrationCase(wdir, TIAMB, ROW_INDICES[wdir])
    sol = calibration.run_model(1, 1, 1, setpoints)
    sim_dict = make_sim_case(sol, wdir, "nocontrol")

    with open(setpoint_fn, "w") as f:
        json.dump(sim_dict, f, indent=4)


def find_optimal_setpoints(
    wdir: float, controller_id: str, calibration_fn: Path | str, setpoint_fn: str | Path
) -> None:
    """
    Calculates the optimal control set points for a given wind direction and
    wake calibration.

    inputs: wind direction (wdir), Wake calibration file (calibration_fn)
    outputs: setpoint JSON file (setpoint_fn)
    """
    with open(calibration_fn, "r") as f:

        out = f.read()
        a, b, c = [float(x) for x in out.split(",")]

    calibration = MultiSimCalibrationCase(wdir, TIAMB, ROW_INDICES[wdir])
    calibration.set_calibration(a, b, c)

    windfarm = calibration.calibrated_windfarm()
    controller = CONTROLLERS[controller_id]

    sol = controller(calibration.layout, windfarm).optimise(
        use_gradients=True, verbose=True
    )

    sim_dict = make_sim_case(sol, wdir, controller_id)
    with open(setpoint_fn, "w") as f:
        json.dump(sim_dict, f, indent=4)


def combine_results(LES_dir: list[Path], out_fn: Path | str) -> None:
    """
    Combines all the simulation output files including LES and MITWindfarm sim results.
    inputs: Path to results (???)
    outputs: Aggregated results dataframe as CSV file (out_fn)
    """
    raise NotImplementedError


def extract_setpoints_from_json(fn: Path | str) -> list[tuple[float, float]]:
    with open(fn, "r") as f:
        _setpoints = json.load(f)

    yaws = np.deg2rad([turbine["yaw"] for turbine in _setpoints["turbines"]])
    ctprimes = [turbine["ctp"] for turbine in _setpoints["turbines"]]
    setpoints = list(zip(ctprimes, yaws))

    return setpoints


def run_MITWindfarm(
    wdir: float,
    controller: str,
    calibration_fn: Path | str,
    setpoint_fn: str | Path,
    res_fn: str | Path,
) -> None:
    # Make output directory if it does not exist.
    with open(calibration_fn, "r") as f:
        out = f.read()
        a, b, c = [float(x) for x in out.split(",")]

    calibration = MultiSimCalibrationCase(wdir, TIAMB, ROW_INDICES[wdir])
    calibration.set_calibration(a, b, c)
    windfarm = calibration.calibrated_windfarm()

    with open(setpoint_fn, "r") as f:
        _setpoints = json.load(f)

    yaws = np.deg2rad([turbine["yaw"] for turbine in _setpoints["turbines"]])
    ctprimes = [turbine["ctp"] for turbine in _setpoints["turbines"]]
    setpoints = list(zip(ctprimes, yaws))

    sol = windfarm(calibration.layout, setpoints)
    print(wdir, controller, sol.Cp)
    _df = utils.to_polars(sol).with_columns(
        pl.lit(controller).alias("controller"),
        pl.lit(wdir).alias("wdir"),
        pl.lit("MITWindfarm").alias("simulator"),
    )

    Path(res_fn).parent.mkdir(exist_ok=True, parents=True)
    _df.write_csv(res_fn)


def plot_layout_single(res_fn: Path | str, fig_fn: Path | str) -> None:
    df = pl.read_csv(res_fn)

    x, y, z = [], [], []
    for _df in df.iter_rows(named=True):
        x.append(_df["x"])
        y.append(_df["y"])
        z.append(_df["z"])
    layout = Layout(x, y, z)

    plot_text_on_layout(
        layout,
        df["Cp"].to_numpy(),
        fig_fn,
        f"{df['Cp'].mean():.3f}",
    )
