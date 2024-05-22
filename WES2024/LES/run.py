import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

import numpy as np
import polars as pl
from mitwindfarm import (
    GaussianWake,
    Layout,
    RotorSolution,
    Square,
    WakeModel,
    Windfarm,
    WindfarmSolution,
)
from scipy.optimize import minimize
from rich import print

from WES2024 import utils
from WES2024.optimise import JointControl, NoControl, ThrustControl, YawControl

LES_output_dir = Path(__file__).parent / "LES_output"
LES_input_dir = Path(__file__).parent / "LES_input"

assert LES_output_dir.exists()
LES_input_dir.mkdir(exist_ok=True, parents=True)

LES_FN_REGEX = re.compile("(\w+)_wdir(-?\d+.\d+)_(\w+).csv")

BASE_LAYOUT = Square(6.0, 5).rotate(45)

TIAMB = 0.053  # Determined from LES.
ROW_INDICES = [
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


class VariableKwGaussianWakeModel(WakeModel):
    def __init__(self, a: float, b: float, c: float, sigma: float = 1 / np.sqrt(8)):
        self.a = a
        self.b = b
        self.c = c
        self.sigma = sigma

    def __call__(self, x, y, z, rotor_sol: "RotorSolution", TIamb: float = None) -> GaussianWake:
        kw = self.a * rotor_sol.TI**2 + self.b * rotor_sol.TI + self.c
        return GaussianWake(x, y, z, rotor_sol, sigma=self.sigma, kw=kw, TIamb=TIamb)


class CalibrationCase:
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
        self._setpoints = None

    def run_model(self, a: float, b: float, c: float) -> WindfarmSolution:
        """
        Run the wind farm model using a variable-wake spreading rate wake model
        for a given set of wake model parameters, (a, b, c).
        """
        wakemodel = VariableKwGaussianWakeModel(a, b, c)
        windfarm = Windfarm(wake_model=wakemodel, TIamb=self.TIamb)

        setpoints = len(self.layout) * [(2.0, 0.0)]
        sol = windfarm(self.layout, setpoints)

        return sol

    def calibrate(
        self, Cp_ref: list[float], x0: tuple[float, float, float] = (0.0, 1.0, 0.0)
    ) -> "CalibrationCase":
        """
        Calibrate a polynomial mapping between TI at the rotor and wake spreading
        wake by minimizing the square error of Cp NORMALIZED by the upstream turbines.
        """

        p_norm_ref = normalize_by_upstream(Cp_ref, row_indices=self.row_indices)

        def func(x):
            a, b, c = x
            sol = self.run_model(a, b, c)
            Cp_model = np.array([x.Cp for x in sol.rotors])
            p_norm = normalize_by_upstream(Cp_model, self.row_indices)

            cost = np.sum((p_norm - p_norm_ref) ** 2)
            return cost

        opt_sol = minimize(func, x0, bounds=[(0, 10), (0, 5), (-1, 1)])

        # print(opt_sol)

        self._setpoints = opt_sol.x
        self.calibrated = True

        return self

    def setpoints(self):
        if not self.calibrated:
            raise ValueError("Case not yet calibrated.")
        else:
            return self._setpoints

    def calibrated_windfarm(self) -> Windfarm:
        a, b, c = self.setpoints()
        wakemodel = VariableKwGaussianWakeModel(a, b, c)
        windfarm = Windfarm(wake_model=wakemodel, TIamb=self.TIamb)

        return windfarm


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


def func(x) -> pl.DataFrame:
    fn = x
    print(f"Calibrating model on {fn.name}...")
    df = pl.read_csv(fn)
    params = extract_fn_params(fn.name)
    wdir = params["wdir"]
    print("calibrating...")
    calibration = CalibrationCase(wdir, TIAMB, ROW_INDICES).calibrate(df["Cp"].to_numpy())

    print(f"calibration: {calibration.setpoints()}")
    windfarm = calibration.calibrated_windfarm()

    out = []
    for controller in [NoControl, ThrustControl, YawControl, JointControl]:
        print(f"Running {controller.__name__} case...")

        # TO DO: get gradients working!
        sol = controller(calibration.layout, windfarm).optimise(use_gradients=False)

        _df = utils.to_polars(sol).with_columns(
            pl.lit(controller.__name__.lower()).alias("controller"),
            pl.lit(wdir).alias("wdir"),
        )

        out.append(_df)

    return pl.concat(out)


def make_sim_case(sol: WindfarmSolution, wdir: float, controller: str) -> dict:

    xmin, ymin = sol.layout.x.min(), sol.layout.y.min()
    casename = f"diamond_wdir{wdir}_{controller}"

    turbines = []
    for i, ((x, y, z), rotor) in enumerate(zip(sol.layout, sol.rotors)):
        _turbine = TurbineDefinition(
            f"turbine_{i}", x - xmin, y - ymin, z, np.rad2deg(rotor.yaw), rotor.Ctprime
        )
        turbines.append(_turbine)

    return asdict(SimulationDefinition(casename, wdir, controller, turbines))


def make_sim_cases(df: pl.DataFrame) -> list[SimulationDefinition]:
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


if __name__ == "__main__":
    # Load LES data which has not yet been calibrated.
    LES_fns = retrieve_LES_output_files(LES_output_dir, LES_input_dir, overwrite=False)

    # Calibrate loaded LES cases.
    dfs = []
    for fn in LES_fns:
        dfs.append(func(fn))

    if len(dfs) == 0:
        sys.exit()

    df = pl.concat(dfs)

    # Save as SimulationDefinition
    cases = make_sim_cases(df)

    # Save setpoints as JSON.
    for simulation in cases:
        sim_dict = asdict(simulation)
        with open(LES_input_dir / f"{simulation.casename}.json", "w") as f:
            json.dump(sim_dict, f, indent=4)

    print("Done.")
