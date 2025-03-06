"""
Starting something new here. Let's compare rotor-by-rotor
estimates of Cp in a "best fit" sense: how well CAN the modeling
framework capture the LES power output? 

We have lots of data to compare against now, let's focus on
the Unified model under all control strategies. 

Kirby Heck
2024 Dec 11
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path
import sys
from WES2024.LES.shared import (
    TIAMB,
    CALIBRATION_LAYOUT,
    normalize_by_upstream,
)

import numpy as np
import polars as pl
from mitwindfarm import (
    GaussianWakeModel,
    VariableKwGaussianWakeModel,
    Layout,
    Square,
    Windfarm,
    WindfarmSolution,
    Niayifar,
    Linear,
)
from scipy.optimize import minimize
from rich import print

from WES2024.CustomRotors import UnifiedLUTAD

LES_output_dir = Path(__file__).parent / "LES_output"
CALIBRATION_RESULTS = Path(__file__).parent / "calibration"# / "thrustcontrol"
CALIBRATION_RESULTS.mkdir(exist_ok=True)

assert LES_output_dir.exists()

LES_FN_REGEX = re.compile("(\w+)_wdir(-?\d+.\d+)_(\w+).csv")

BASE_LAYOUT = Square(6.0, 5).rotate(45).rotate(-2.5)

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

ROW_INDICES_CALIBRATION = np.arange(54).reshape(18, 3)

# for now use Unified
ROTOR_MODEL = UnifiedLUTAD()


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
        params = extract_fn_params(file.name)
        df = pl.read_csv(file)
        ret.append(df.with_columns([pl.lit(val).alias(key) for key, val in params.items()]))

    # concatenate list and return
    return pl.concat(ret)

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

        sol = self.run_windfarm(x, control_setpoints)

        Cp_model = np.array([x.Cp for x in sol.rotors])
        # p_norm = normalize_by_upstream(Cp_model, self.row_indices)
        # p_norm_ref = normalize_by_upstream(p_ref, self.row_indices)
        p_norm = Cp_model / (16/27)  # normalize by Betz
        p_norm_ref = p_ref / LES_pnormfact

        cost = np.linalg.norm(p_norm - p_norm_ref, ord=2) / np.sqrt(len(p_norm))
        # cost = sum((p_norm - p_norm_ref)**2)  # sum of squares
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
        return x


class Calibratekw_niayifar(Calibration):
    def initial_guess(self) -> tuple[float]:
        return 0.07

    def bounds(self) -> list[tuple[float]]:
        return [(0, 1.0)]

    def run_windfarm(self, x, control_setpoints, superposition=Niayifar()) -> WindfarmSolution:
        kw = x
        wakemodel = GaussianWakeModel(sigma=1/np.sqrt(8), kw=kw, x0=3)
        windfarm = Windfarm(
            rotor_model=UnifiedLUTAD(),
            wake_model=wakemodel,
            TIamb=TIAMB,
            superposition=superposition,
        )

        return windfarm(self.layout, control_setpoints)


class Calibratekw_FLS(Calibratekw_niayifar):
    def run_windfarm(self, x, control_setpoints, superposition=Linear()) -> WindfarmSolution:
        kw = x
        wakemodel = GaussianWakeModel(sigma=1 / np.sqrt(8), kw=kw, x0=3)
        windfarm = Windfarm(
            rotor_model=UnifiedLUTAD(),
            wake_model=wakemodel,
            TIamb=TIAMB,
            superposition=superposition,
        )

        return windfarm(self.layout, control_setpoints)


class Calibratekw_x01_niayifar(Calibratekw_niayifar):
    def run_windfarm(self, x, control_setpoints, superposition=Niayifar()) -> WindfarmSolution:
        kw = x
        wakemodel = GaussianWakeModel(sigma=1 / np.sqrt(8), kw=kw, x0=1)
        windfarm = Windfarm(
            rotor_model=UnifiedLUTAD(),
            wake_model=wakemodel,
            TIamb=TIAMB,
            superposition=superposition,
        )

        return windfarm(self.layout, control_setpoints)


class CalibrateGaussian_niayifar(Calibration):
    def initial_guess(self) -> tuple[float]:
        return 0.07, 0.25

    def bounds(self) -> list[tuple[float]]:
        return [(0, 1.0), (0, 1)]

    def run_windfarm(self,x,control_setpoints,superposition=Niayifar(),) -> WindfarmSolution:
        kw, sigma = x
        wakemodel = GaussianWakeModel(sigma=sigma, kw=kw, x0=3)
        windfarm = Windfarm(
            rotor_model=UnifiedLUTAD(),
            wake_model=wakemodel,
            TIamb=TIAMB,
            superposition=superposition,
        )

        return windfarm(self.layout, control_setpoints)


class CalibrateLinear_FLS(Calibration):
    def initial_guess(self) -> tuple[float]:
        return 1.0, 0., 0.0

    def bounds(self) -> list[tuple[float]]:
        return [(0, 5), (-5, 5), (-1, 1)]

    def run_windfarm(self, x, control_setpoints) -> WindfarmSolution:
        a, b, c = x
        wakemodel = VariableKwGaussianWakeModel(a, 0, c, x0=3, )
        windfarm = Windfarm(rotor_model=UnifiedLUTAD(), wake_model=wakemodel, TIamb=TIAMB)

        return windfarm(self.layout, control_setpoints)


class CalibrateLinear_niayifar(Calibration):
    def initial_guess(self) -> tuple[float]:
        return 1.0, 0., 0.0

    def bounds(self) -> list[tuple[float]]:
        return [(0, 5), (-5, 5), (-1, 1)]

    def run_windfarm(self, x, control_setpoints) -> WindfarmSolution:
        a, b, c = x
        wakemodel = VariableKwGaussianWakeModel(a, 0, c, x0=3, )
        windfarm = Windfarm(rotor_model=UnifiedLUTAD(), wake_model=wakemodel, TIamb=TIAMB, superposition=Niayifar())

        return windfarm(self.layout, control_setpoints)


class CalibrateLinearCtprime_FLS(CalibrateLinear_FLS):
    """Linear calibration + C_T' term"""

    def run_windfarm(self, x, control_setpoints) -> WindfarmSolution:
        a, b, c = x
        wakemodel = VariableKwGaussianWakeModel(a, b, c, x0=3, )
        windfarm = Windfarm(rotor_model=UnifiedLUTAD(), wake_model=wakemodel, TIamb=TIAMB)

        return windfarm(self.layout, control_setpoints)


class CalibrateLinearCtprime_niayifar(CalibrateLinear_FLS):
    """Linear calibration + C_T' term, Niayifar superposition"""

    def run_windfarm(self, x, control_setpoints) -> WindfarmSolution:
        a, b, c = x
        wakemodel = VariableKwGaussianWakeModel(a, b, c, x0=3, )
        windfarm = Windfarm(rotor_model=UnifiedLUTAD(), wake_model=wakemodel, TIamb=TIAMB, superposition=Niayifar())

        return windfarm(self.layout, control_setpoints)


wakemodels = {
    "00_kw_niayifar": Calibratekw_niayifar,
    "01_kw_FLS": Calibratekw_FLS,
    "02_kw_x0_1D": Calibratekw_x01_niayifar,
    "03_kw_sigma0": CalibrateGaussian_niayifar,
    "04_kw_TI": CalibrateLinear_niayifar,
    "05_kw_TI_FLS": CalibrateLinear_FLS,
    "06_kw_TI_Ctp": CalibrateLinearCtprime_niayifar,
    "07_kw_TI_Ctp_FLS": CalibrateLinearCtprime_FLS,
}


def run(regenerate=False, methods=None, fname="apriori_err.json"): 
    """Run a priori calibration tests"""
    # load LES data
    df = read_LES_outputs(LES_output_dir.glob("*.csv"))

    # select a few methods to run comparison
    methods = methods or ["nocontrol", "calibration_18x3", "thrustcontrol"]
    df = df.filter(pl.col("method").is_in(methods))

    # (try to) load cached results
    try: 
        df_cached = pl.read_json(CALIBRATION_RESULTS / fname)
    except FileNotFoundError: 
        regenerate = True

    ret = []
    for method, _df in df.group_by("method"):
        print("Starting control case: ", method)

        for key, wakemodel in wakemodels.items():
            # check if we already ran this case
            if not regenerate and len(df_cached.filter(method=method, wakemodel=key)) > 0: 
                continue  # skip - already computed
            
            # Setup MITWindfarm
            if method == "calibration_18x3": 
                calibration = wakemodel(CALIBRATION_LAYOUT, ROW_INDICES_CALIBRATION)
            else: 
                calibration = wakemodel(BASE_LAYOUT, ROW_INDICES)
            control_setpoints = _df.select("Ctprime", "yaw").to_numpy()  # set setpoints

            # calibrate wake model parameters
            calib = calibration.calibrate(_df["Cp"].to_numpy(), control_setpoints)
            # run MITWindfarm
            sol = calibration.run_windfarm(calib, control_setpoints)

            # compute error
            cost = calibration.cost(calib, _df["Cp"].to_numpy(), control_setpoints)
            ret.append(dict(method=method, wakemodel=key, err=cost, Cp_farm=sol.Cp, params=list(calib)))
            print(f"  Done with {key}")
        print("  - Done with control case: ", method)

    # cast to dataframe
    df_out = pl.DataFrame(ret)
    if not regenerate:  # concatenate to existing rather than overwriting
        df_out = pl.concat([df_out, df_cached]).sort(by=['method', 'wakemodel'])
    print(df_out)

    # save results to csv, json
    print(f"Saving results to {CALIBRATION_RESULTS}")
    df_out.write_json(CALIBRATION_RESULTS / fname, pretty=True)
    df_out.select(pl.col("*").exclude("params")).write_csv(CALIBRATION_RESULTS / "apriori_err.csv")


if __name__ == "__main__":
    run(regenerate=False)