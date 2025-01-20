# dualitic must be imported first
# from dualitic import DualVariables

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from mitwindfarm import (
    GaussianWake,
    Layout,
    RotorSolution,
    VariableKwGaussianWakeModel,
    WakeModel,
    Windfarm,
    WindfarmSolution,
    Niayifar,
)
from rich import print
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

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


LES_INPUT_JSON_FN = STEP_1_DIR / "calibration_diamond.json"
LES_OUTPUT_FN = STEP_1_DIR / "LES_wdir-2.5_nocontrol.csv"
CALIBRATION_FN = STEP_2_DIR / "calibration.csv"
CALIBRATION_FN2 = STEP_2_DIR / "calibration_opt.csv"
FIG_FN = STEP_2_DIR / "calibration_results.png"
FIG_DATA_FN = STEP_2_DIR / "calibration_results_data.csv"
CACHE_FN = STEP_2_DIR / "cache.csv"


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

    def __call__(self, x, y, z, rotor_sol: RotorSolution, TIamb: float = None) -> GaussianWake:
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
            x0=3.0,
        )


# class VariableKwGaussianWakeModel2(WakeModel):
#     def __init__(
#         self,
#         a: float,
#         b: float,
#         c: float,
#         d: float,
#         sigma: float = 1 / np.sqrt(8),
#         WATI_sigma_multiplier=1.0,
#         xmax: float = 100.0,
#     ):
#         self.a = a
#         self.b = b
#         self.c = c
#         self.d = d
#         self.sigma = sigma
#         self.xmax = xmax
#         self.WATI_sigma_multiplier = WATI_sigma_multiplier

#     def __call__(self, x, y, z, rotor_sol: "RotorSolution", TIamb: float = None) -> GaussianWake:
#         kw = (
#             self.a * rotor_sol.TI
#             + self.b * rotor_sol.Ctprime
#             + self.c * rotor_sol.TI * rotor_sol.Ctprime
#             + self.d
#         )
#         return GaussianWake(
#             x,
#             y,
#             z,
#             rotor_sol,
#             sigma=self.sigma,
#             kw=kw,
#             TIamb=TIamb,
#             xmax=self.xmax,
#             WATI_sigma_multiplier=self.WATI_sigma_multiplier,
#         )


class Calibration(ABC):
    def __init__(self, layout: Layout, row_indices: list[list[float]]):
        self.layout = layout
        self.row_indices = row_indices

    @abstractmethod
    def initial_guess(self) -> tuple[float]:
        ...

    @abstractmethod
    def run_windfarm(self, x, control_setpoints) -> WindfarmSolution:
        ...

    @abstractmethod
    def bounds(self) -> list[tuple[float]]:
        ...

    def cost(self, x, p_norm_ref, control_setpoints) -> float:
        sol = self.run_windfarm(x, control_setpoints)

        Cp_model = np.array([x.Cp for x in sol.rotors])
        p_norm = normalize_by_upstream(Cp_model, self.row_indices)

        cost = np.sum((p_norm - p_norm_ref) ** 2)

        return cost

    def calibrate(
        self, powers: list[float], control_setpoints: list[tuple[float, float]]
    ) -> list[float]:
        p_norm_ref = normalize_by_upstream(powers, self.row_indices)

        opt_sol = minimize(
            self.cost,
            self.initial_guess(),
            bounds=self.bounds(),
            args=(p_norm_ref, control_setpoints),
            options={"disp": True},
        )

        print(opt_sol)

        return self.postproc(opt_sol.x)

    def postproc(self, x):
        return x


class CalibrateLinear(Calibration):
    def initial_guess(self) -> tuple[float]:
        return 1.0, 1.0, 0.0

    def bounds(self) -> list[tuple[float]]:
        return [(0, 5), (-5, 5), (-1, 1)]

    def run_windfarm(self, x, control_setpoints) -> WindfarmSolution:
        a, b, c = x
        windfarm = Windfarm(
            rotor_model=UnifiedLUTAD(),
            wake_model=VariableKwGaussianWakeModel(a, b, c, x0=3.0),
            TIamb=TIAMB,
            superposition=Niayifar(),
        )

        return windfarm(self.layout, control_setpoints)


# class CalibrateLinearFirstRow(CalibrateLinear):
#     """
#     Calibrates the linear kw model considering the power of the second row
#     turbines only (i.e. the effect of the first row on their immediate
#     downstream turbines).
#     """

#     def cost(self, x, p_norm_ref, control_setpoints) -> float:
#         # upstream_turbine_indices = [row[1] for row in self.row_indices if len(row) > 1]
#         most_upstream = [row[0] for row in self.row_indices]
#         downstream = [x for x in range(len(self.layout)) if x not in most_upstream]

#         sol = self.run_windfarm(x, control_setpoints)

#         Cp_model = np.array([x.Cp for x in sol.rotors])
#         p_norm = normalize_by_upstream(Cp_model, self.row_indices)

#         cost = np.sum((p_norm[downstream] - p_norm_ref[downstream]) ** 2)

#         return cost


# class CalibrateLinearFirstRow2(Calibration):
#     """
#     Calibrates the linear kw model considering the power of the second row
#     turbines only (i.e. the effect of the first row on their immediate
#     downstream turbines).
#     """

#     def initial_guess(self) -> tuple[float]:
#         return 1.21, 0.027, -0.21, -0.027  # from manual calibration

#     def bounds(self) -> list[tuple[float]]:
#         return [(0, 5), (-5, 5), (-1, 1), (-1, 1)]

#     def run_windfarm(self, x, control_setpoints) -> WindfarmSolution:
#         a, b, c, d = x
#         wakemodel = VariableKwGaussianWakeModel2(a, b, c, d)
#         windfarm = Windfarm(
#             rotor_model=UnifiedLUTAD(), wake_model=wakemodel, TIamb=TIAMB, superposition=Niayifar()
#         )

#         return windfarm(self.layout, control_setpoints)

#     def cost(self, x, p_norm_ref, control_setpoints) -> float:
#         # upstream_turbine_indices = [row[1] for row in self.row_indices if len(row) > 1]
#         most_upstream = [row[0] for row in self.row_indices]
#         downstream = [x for x in range(len(self.layout)) if x not in most_upstream]

#         sol = self.run_windfarm(x, control_setpoints)

#         Cp_model = np.array([x.Cp for x in sol.rotors])
#         p_norm = normalize_by_upstream(Cp_model, self.row_indices)

#         cost = np.sum((p_norm[downstream] - p_norm_ref[downstream]) ** 2)

#         return cost


class CalibrateIndividual(Calibration):
    """
    Calibrates kw for each turbine individually.
    """

    def initial_guess(self) -> tuple[float]:
        return [0.07 for _ in self.layout]

    def bounds(self) -> list[tuple[float]]:
        return [(0, 1) for _ in self.layout]

    def run_windfarm(self, x: list[float], control_setpoints) -> WindfarmSolution:
        wakemodel = CustomKwWakeModel(x)
        windfarm = Windfarm(
            rotor_model=UnifiedLUTAD(), wake_model=wakemodel, TIamb=TIAMB, superposition=Niayifar()
        )

        return windfarm(self.layout, control_setpoints)


# class CalibrateManual(Calibration):
#     """
#     Manual calibration of the model.
#     """

#     def initial_guess(self) -> tuple[float]:
#         ...

#     def bounds(self) -> list[tuple[float]]:
#         ...

#     def run_windfarm(self, x: list[float], control_setpoints) -> WindfarmSolution:
#         a, b, c, d = x
#         wakemodel = VariableKwGaussianWakeModel2(a, b, c, d)
#         windfarm = Windfarm(
#             rotor_model=UnifiedLUTAD(), wake_model=wakemodel, TIamb=TIAMB, superposition=Niayifar()
#         )

#         return windfarm(self.layout, control_setpoints)

#     def calibrate(*args, **kwargs):
#         return 1.5, 0.027, -0.21, -0.04


class NoCalibration(Calibration):
    """
    Equivalent to the fixed kw model
    """

    def initial_guess(self) -> tuple[float]:
        ...

    def bounds(self) -> list[tuple[float]]:
        ...

    def run_windfarm(self, x: list[float], control_setpoints) -> WindfarmSolution:
        a, b, c = x
        wakemodel = VariableKwGaussianWakeModel(a, b, c)
        windfarm = Windfarm(
            rotor_model=UnifiedLUTAD(), wake_model=wakemodel, TIamb=TIAMB, superposition=Niayifar()
        )

        return windfarm(self.layout, control_setpoints)

    def calibrate(*args, **kwargs):
        return 0.0, 0.0, 0.07


def sol_to_polars(sol: WindfarmSolution) -> pl.DataFrame:
    df = pl.from_dict(
        {
            "kw": [x.kw for x in sol.wakes],
            "Cp": [x.Cp for x in sol.rotors],
            "an": [x.an for x in sol.rotors],
            "Ct": [x.Ct for x in sol.rotors],
            "Ctprime": [x.Ctprime for x in sol.rotors],
            "TI": [x.TI for x in sol.rotors],
            "idx": [x.idx for x in sol.rotors],
        }
    )

    return df


calibrations: dict[str, Calibration] = {
    "CalibrateIndividual": CalibrateIndividual,
    "CalibrateLinear": CalibrateLinear,
    "NoCalibration": NoCalibration,
}


def run(
    LES_output_fns: Path, case_json_fn: Path, cache: Optional[pl.DataFrame] = None
) -> pl.DataFrame:
    les_powers = pl.read_csv(LES_output_fns)["Cp"].to_numpy()
    control_setpoints = SimulationDefinition.from_json(case_json_fn).setpoints()

    df_list = []
    for name, calibration in calibrations.items():
        print(name)
        if cache is not None and name == "CalibrateIndividual":
            df_list.append(cache.filter(calib_method=name))
        else:
            calibration = calibration(CALIBRATION_LAYOUT, ROW_INDICES)
            calib = calibration.calibrate(les_powers, control_setpoints)
            sol = calibration.run_windfarm(calib, control_setpoints)
            df = sol_to_polars(sol).with_columns(pl.lit(name).alias("calib_method"))
            df_list.append(df)

            if name == "CalibrateLinear":
                with open(CALIBRATION_FN, "w") as f:
                    f.write(",".join([str(x) for x in calib]))

            if name == "CalibrateLinearOpt":
                with open(CALIBRATION_FN2, "w") as f:
                    f.write(",".join([str(x) for x in calib]))

    df = pl.concat(df_list)
    return df


if __name__ == "__main__":
    if not CACHE_FN.exists():
        # _df = pl.read_csv(CACHE_FN)
        df = run(LES_OUTPUT_FN, LES_INPUT_JSON_FN)
        df.write_csv(CACHE_FN)
    else:
        _df = pl.read_csv(CACHE_FN)
        df = run(LES_OUTPUT_FN, LES_INPUT_JSON_FN, cache=_df)

    # remove last turbines
    last_turbine_idxs = [row[-1] for row in ROW_INDICES]

    most_upstream = [row[0] for row in ROW_INDICES]
    second_upstream = [row[1] for row in ROW_INDICES if len(row) >= 2]
    last_upstream = [row[-1] for row in ROW_INDICES]

    n_upstream = []
    for idx in df["idx"]:
        if idx in most_upstream:
            n_upstream.append(0)
        elif idx in second_upstream:
            n_upstream.append(1)
        elif idx in last_upstream:
            n_upstream.append(2)
        else:
            n_upstream.append(-1)

    df = df.with_columns(pl.Series(n_upstream).alias("n_upstream"))

    df.write_csv(FIG_DATA_FN)
    fig, axes = plt.subplots(1, 2, figsize=5 * np.array([2, 1]), sharey=True)

    sns.scatterplot(
        df.filter(pl.col("calib_method") != "CalibrateLinear"),
        x="TI",
        y="kw",
        hue="calib_method",
        style="n_upstream",
        ax=axes[0],
        legend=False,
    )
    sns.scatterplot(df, x="Ctprime", y="kw", hue="calib_method", style="n_upstream", ax=axes[1])

    axes[0].set_ylabel(r"$k_w$")
    axes[0].set_xlabel(r"$TI$")
    axes[1].set_xlabel(r"$C_T'$")
    plt.savefig(FIG_FN, dpi=300, bbox_inches="tight")
