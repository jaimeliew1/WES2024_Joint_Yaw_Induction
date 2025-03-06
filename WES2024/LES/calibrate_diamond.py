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


LES_INPUT_JSON_FN = STEP_1_DIR / "calibration_18x3.json"
LES_OUTPUT_FN = STEP_1_DIR / "LES_wdir-2.5_calibration_18x3.csv"
CALIBRATION_FN = STEP_2_DIR / "calibration.csv"
CALIBRATION_FN2 = STEP_2_DIR / "calibration_opt.csv"
FIG_FN = STEP_2_DIR / "calibration_results.png"
CACHE_FN = STEP_2_DIR / "cache.csv"

# ROW_INDICES = [
#     [0, 1, 2],
#     [3, 4, 5],
#     [6, 7, 8],
#     [9, 10, 11],
#     [12, 13, 14],
#     [15, 16, 17],
#     [18, 19, 20],
#     [21, 22, 23],
#     [24, 25, 26],
#     [27, 28, 29],
#     [30, 31, 32],
#     [33, 34, 35],
#     [36, 37, 38],
#     [39, 40, 41],
#     [42, 43, 44],
#     [45, 46, 47],
#     [48, 49, 50],
#     [51, 52, 53],
# ]


# >==< >==< >==< TEST PARAMETERS. REMOVE WHEN KIRBY HAS NEW DATA >==< >==< >==<
LES_INPUT_JSON_FN = (
    Path(__file__).parent / "debug" / "wdir-2.5_fixedkw_nocontrol.json"
)
LES_OUTPUT_FN = (
    Path(__file__).parent / "debug" / "LES_wdir-2.5_nocontrol.csv"
)

from mitwindfarm import Square

CALIBRATION_LAYOUT = Square(6.0, 5).rotate(45)

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

# >==< >==< >==< END TEST PARAMETERS >==< >==< >==<


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
        )


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
        windfarm = Windfarm(rotor_model=UnifiedLUTAD(), wake_model=wakemodel, TIamb=TIAMB)

        return windfarm(self.layout, control_setpoints)


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


calibrations = {
    "CalibrateIndividual": CalibrateIndividual,
    # "CalibrateLinearOpt": CalibrateLinearFirstRow2,
    # "CalibrateLinear": CalibrateLinear,
    # "CalibrateLinearFirstRow": CalibrateLinearFirstRow,
}


def run(LES_output_fns: Path, case_json_fn: Path, cache: Optional[pl.DataFrame] = None):

    les_powers = pl.read_csv(LES_output_fns)["Cp"].to_numpy()
    control_setpoints = SimulationDefinition.from_json(case_json_fn).setpoints()

    df_list = []
    for name, calibration in calibrations.items():
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
    second_upstream = [row[1] for row in ROW_INDICES]
    last_upstream = [row[2] for row in ROW_INDICES]

    n_upstream = []
    for idx in df["idx"]:
        if idx in most_upstream:
            n_upstream.append(0)
        elif idx in second_upstream:
            n_upstream.append(1)
        elif idx in last_upstream:
            n_upstream.append(2)
        else:
            raise ValueError()

    df = df.with_columns(pl.Series(n_upstream).alias("n_upstream"))

    fig, axes = plt.subplots(1, 2, figsize=5 * np.array([2, 1]), sharey=True)

    sns.scatterplot(
        df,
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
