# dualitic must be imported first
# from dualitic import DualVariables

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import polars as pl
from mitwindfarm import (
    GaussianWake,
    Layout,
    RotorSolution,
    VariableKwGaussianWakeModel,
    WakeModel,
    Windfarm,
)
from rich import print
from scipy.optimize import minimize

from WES2024.CustomRotors import UnifiedLUTAD
from WES2024.LES.shared import (
    CALIBRATION_LAYOUT,
    STEP_1_DIR,
    STEP_2_DIR,
    TIAMB,
    SimulationDefinition,
    normalize_by_upstream,
)

### Parameters
# File paths


LES_INPUT_JSON_FN = STEP_1_DIR / "calibration_18x3.json"
LES_OUTPUT_FN = STEP_2_DIR / "???"
CALIBRATION_FN = STEP_2_DIR / "calibration.csv"
FIG_FN = STEP_2_DIR / "calibration_results.png"


ROW_INDICES = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 10, 11],
    [12, 13, 14],
    [15, 16, 17],
    [18, 19, 20],
    [21, 22, 23],
    [24, 25, 26],
    [27, 28, 29],
    [30, 31, 32],
    [33, 34, 35],
    [36, 37, 38],
    [39, 40, 41],
    [42, 43, 44],
    [45, 46, 47],
    [48, 49, 50],
    [51, 52, 53],
]


class CustomKwWakeModel(WakeModel):
    def __init__(
        self,
        kws: list[float],
        sigma=0.25,
        WATI_sigma_multiplier=1.0,
        xmax: float = 100.0,
    ):
        self.sigma = sigma
        self.kws = kws
        self.xmax = xmax
        self.WATI_sigma_multiplier = WATI_sigma_multiplier

    def __call__(
        self, x, y, z, rotor_sol: RotorSolution, TIamb: float = None
    ) -> GaussianWake:
        return GaussianWake(
            x,
            y,
            z,
            rotor_sol,
            sigma=self.sigma,
            kw=self.kws[rotor_sol.idx],
            TIamb=TIamb,
            xmax=self.xmax,
            WATI_sigma_multiplier=self.WATI_sigma_multiplier,
        )


class Calibration(ABC):
    @abstractmethod
    def initial_guess(self) -> tuple[float]:
        ...

    @abstractmethod
    def cost(self, x, p_norm_ref, control_setpoints) -> float:
        ...

    @abstractmethod
    def bounds(self) -> list[tuple[float]]:
        ...

    def calibrate(
        self, powers: list[float], control_setpoints: list[tuple[float, float]]
    ) -> list[float]:
        p_norm_ref = normalize_by_upstream(powers, self.row_indices)

        opt_sol = minimize(
            self.cost,
            self.initial_guess(),
            bounds=self.bounds(),
            args=(p_norm_ref, control_setpoints),
        )

        print(opt_sol)

        return opt_sol.x


class CalibrateLinear(Calibration):
    def __init__(self, layout: Layout, row_indices: list[list[float]]):
        self.layout = layout
        self.row_indices = row_indices

    def initial_guess(self) -> tuple[float]:
        return 1.0, 1.0, 0.0

    def bounds(self) -> list[tuple[float]]:
        return [(0, 5), (-5, 5), (-1, 1)]

    def _make_windfarm(self, a, b, c) -> Windfarm:
        wakemodel = VariableKwGaussianWakeModel(a, b, c)
        windfarm = Windfarm(
            rotor_model=UnifiedLUTAD(), wake_model=wakemodel, TIamb=TIAMB
        )

        return windfarm

    def cost(self, x, p_norm_ref, control_setpoints) -> float:
        a, b, c = x

        windfarm = self._make_windfarm(a, b, c)
        sol = windfarm(self.layout, control_setpoints)

        Cp_model = np.array([x.Cp for x in sol.rotors])
        p_norm = normalize_by_upstream(Cp_model, self.row_indices)

        cost = np.sum((p_norm - p_norm_ref) ** 2)

        return cost


class CalibrateLinearFirstRow(CalibrateLinear):
    """
    Calibrates the linear kw model considering the power of the second row
    turbines only (i.e. the effect of the first row on their immediate
    downstream turbines).
    """

    def cost(self, x, p_norm_ref, control_setpoints) -> float:
        upstream_turbine_indices = [row[1] for row in self.row_indices]

        a, b, c = x
        windfarm = self._make_windfarm(a, b, c)
        sol = windfarm(self.layout, control_setpoints)

        Cp_model = np.array([x.Cp for x in sol.rotors])
        p_norm = normalize_by_upstream(Cp_model, self.row_indices)

        cost = np.sum(
            (p_norm[upstream_turbine_indices] - p_norm_ref[upstream_turbine_indices])
            ** 2
        )

        return cost


class CalibrateIndividual(Calibration):
    """
    Calibrates kw for each turbine individually.
    """

    def __init__(self, layout: Layout, row_indices: list[list[float]]):
        self.layout = layout
        self.row_indices = row_indices

    def initial_guess(self) -> tuple[float]:
        return (0.07 for _ in self.layout)

    def bounds(self) -> list[tuple[float]]:
        return [(0, 1) for _ in self.layout]

    def _make_windfarm(self, kws: list[float]) -> Windfarm:
        wakemodel = CustomKwWakeModel(kws)
        windfarm = Windfarm(
            rotor_model=UnifiedLUTAD(), wake_model=wakemodel, TIamb=TIAMB
        )

        return windfarm

    def cost(self, x, p_norm_ref, control_setpoints) -> float:
        kws = x

        windfarm = self._make_windfarm(kws)
        sol = windfarm(self.layout, control_setpoints)

        Cp_model = np.array([x.Cp for x in sol.rotors])
        p_norm = normalize_by_upstream(Cp_model, self.row_indices)

        cost = np.sum((p_norm - p_norm_ref) ** 2)

        return cost


def run(LES_output_fns: Path, case_json_fn: Path):

    les_powers = pl.read_csv(LES_output_fns)["Cp"].to_numpy()
    control_setpoints = SimulationDefinition.from_json(case_json_fn).setpoints()

    calib1 = CalibrateIndividual(CALIBRATION_LAYOUT, ROW_INDICES).calibrate(
        les_powers, control_setpoints
    )
    calib2 = CalibrateLinear(CALIBRATION_LAYOUT, ROW_INDICES).calibrate(
        les_powers, control_setpoints
    )
    calib3 = CalibrateLinearFirstRow(CALIBRATION_LAYOUT, ROW_INDICES).calibrate(
        les_powers, control_setpoints
    )

    print(calib1, calib2, calib3)


if __name__ == "__main__":
    run(LES_OUTPUT_FN, LES_INPUT_JSON_FN)
