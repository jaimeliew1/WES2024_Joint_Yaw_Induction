"""
Final iteration of LES calibration using the
new no control LES data to calibrate a gaussian wake model
with Niayifar superposition.

Kirby Heck
2025 March 07
"""

import re
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import polars as pl
from mitwindfarm import (
    VariableKwGaussianWakeModel,
    Layout,
    Square,
    Windfarm,
    WindfarmSolution,
    Niayifar,
)
from scipy.optimize import minimize

from WES2024.CustomRotors import UnifiedLUTAD
from WES2024.utils import ROW_INDICES
from WES2024.LES.shared import TIAMB

LES_output_dir = Path(__file__).parent / "LES_output" / "iter_01"
CALIBRATION_RESULTS = Path(__file__).parent / "calibration"
CALIBRATION_RESULTS.mkdir(exist_ok=True)
default_fname = "final_calibration.json"
LES_FN_REGEX = re.compile("(\w+)_wdir(-?\d+.\d+)_(\w+).csv")

BASE_LAYOUT = Square(6.0, 5).rotate(45).rotate(-2.5)
ROTOR_MODEL = UnifiedLUTAD()

x0 = 1  # diameters
sigma = 1 / np.sqrt(8)  # sigma_0


def extract_fn_params(fn: str) -> dict:
    """
    Extracts simulation type, wind direction and control method from filename of
    LES output data.
    """
    groups = LES_FN_REGEX.match(fn)
    if groups is None:
        raise ValueError("Could not match filename regex.")
    groups = groups.groups()
    out = dict(simulator=groups[0], wdir=float(groups[1]), method=groups[2])
    return out


def read_LES_outputs(filepaths):
    """
    Read LES output files and return a dataframe
    """

    ret = []
    for file in filepaths:
        try:
            params = extract_fn_params(file.name)
        except ValueError:
            # skip these .csv files
            continue
        df = pl.read_csv(file)
        ret.append(df.with_columns([pl.lit(val).alias(key) for key, val in params.items()]))

    # concatenate list and return
    return pl.concat(ret, how="diagonal_relaxed")


# Compute LES Cp normalizing factor (Equivalent of Betz)
LES_pnormfact = (
    read_LES_outputs(list(LES_output_dir.glob("*LESnew_nocontrol.csv")))
    .filter(turbine_id=20)["Cp"]
    .item()
)


class Calibration(ABC):
    """Abstract base class for wake model calibration"""

    def __init__(self, layout: Layout, row_indices: list[list[float]]):
        self.layout = layout
        self.row_indices = row_indices

    @abstractmethod
    def initial_guess(self) -> tuple[float]: ...

    @abstractmethod
    def run_windfarm(self, x, control_setpoints) -> WindfarmSolution: ...

    @abstractmethod
    def bounds(self) -> list[tuple[float]]: ...

    def cost(self, x, p_ref, control_setpoints) -> float:
        """Cost function normalizes model and LES Cp and then computes RMSE"""

        sol = self.run_windfarm(x, control_setpoints)

        Cp_model = np.array([x.Cp for x in sol.rotors])
        p_norm = Cp_model / (16 / 27)  # normalize by Betz
        p_norm_ref = p_ref / LES_pnormfact

        # RMSE:
        cost = np.linalg.norm(p_norm - p_norm_ref, ord=2) / np.sqrt(len(p_norm))
        return cost

    def calibrate(
        self, powers: list[float], control_setpoints: list[tuple[float, float]]
    ) -> list[float]:

        opt_sol = minimize(
            self.cost,
            self.initial_guess(),
            bounds=self.bounds(),
            args=(powers, control_setpoints),
            options={"disp": False},
        )
        return self.postproc(opt_sol.x)

    def postproc(self, x):
        """Default postprocessing is no postprocessing"""
        return x


class CalibrateLinear_niayifar(Calibration):
    def initial_guess(self) -> tuple[float]:
        return 1.0, 0.0, 0.0

    def bounds(self) -> list[tuple[float]]:
        return [(0, 5), (-5, 5), (0, 1)]

    def run_windfarm(self, x, control_setpoints) -> WindfarmSolution:
        a, b, c = x
        wakemodel = VariableKwGaussianWakeModel(
            a,
            0,
            c,
            x0=x0,
            sigma=sigma,
        )
        windfarm = Windfarm(
            rotor_model=UnifiedLUTAD(), wake_model=wakemodel, TIamb=TIAMB, superposition=Niayifar()
        )

        return windfarm(self.layout, control_setpoints)


def run(regenerate=False, sim_name="LESnew_nocontrol", fname=default_fname):
    """Run a priori calibration tests"""
    if not regenerate and (CALIBRATION_RESULTS / fname).exists():
        print("File already exists, pass regenerate=True to overwrite.")
        with open(CALIBRATION_RESULTS / default_fname, "r") as f:
            return json.load(f)["params"]

    # load (all) LES data
    df = read_LES_outputs(LES_output_dir.glob("*.csv"))

    # select only the no_control data methods to run comparison
    df = df.filter(pl.col("method") == sim_name)

    # Setup MITWindfarm
    calibration = CalibrateLinear_niayifar(BASE_LAYOUT, ROW_INDICES)
    control_setpoints = df.select("Ctprime", "yaw").to_numpy()  # set setpoints

    # calibrate wake model parameters
    calib = calibration.calibrate(df["Cp"].to_numpy(), control_setpoints)
    params = {a: val for a, val in zip(["a", "b", "c"], calib)}
    params.update(x0=x0, sigma=sigma)
    # run MITWindfarm
    sol = calibration.run_windfarm(calib, control_setpoints)

    # compute error
    cost = calibration.cost(calib, df["Cp"].to_numpy(), control_setpoints)

    # compile results and write to .json
    ret = dict(
        calibration_sim=sim_name, wakemodel="04_kw_TI", err=cost, Cp_farm=sol.Cp, params=params
    )
    print(f"Saving results to {CALIBRATION_RESULTS / fname}")
    with open(CALIBRATION_RESULTS / fname, "w") as f:
        json.dump(ret, f, indent=4)

    return ret["params"]  # return calibration parameters


def get_calibration_params():
    """Retrieves final calibration parameters for Niayifar wake model"""
    if (CALIBRATION_RESULTS / default_fname).exists():
        with open(CALIBRATION_RESULTS / default_fname, "r") as f:
            params = json.load(f)["params"]
        return params
    else:
        return run()


def get_wakemodel():
    """Returns gaussian wake model with the final calibration parameters"""
    return VariableKwGaussianWakeModel(**get_calibration_params())


if __name__ == "__main__":
    run(regenerate=True)
