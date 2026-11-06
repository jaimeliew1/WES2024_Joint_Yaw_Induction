"""
R1: which farm-LES run pairs with which set points / wake-model prediction (the 5x5 aligned farm, wdir 0, 6 D spacing).

Used by the generate/ scripts (LES power cache, row-wise model power cache) and the forpaper/ figures of the farm-control study
(setpoint schematics, LES power gain, turbine-level power). Every set-point CSV is an optimiser output of
generate_les_setpoints_tiled5x5.py (data/generate_les_setpoints_tiled5x5/tiled/<stem>.csv): the 5x1 column optimum tiled over
the 5 columns and re-evaluated as a 5x5 farm, with the model's own Cp per turbine and Cp_farm. The LES directories under LES_BASE
were run with exactly these set points (their setpoints.csv is a copy of the same rows).

TANDEM is always the version with the ground (`tandemwall_bem_*`; bottom_wall_z = -1), which is also how it is run in every figure.

CASES keys: "NoControl", "Thrust_TANDEM", "Yaw_TANDEM", "YawMod_TANDEM", "Joint_TANDEM", "Joint_Vortex", "Joint_Gauss".
"""

from pathlib import Path

import numpy as np
import pandas as pd
from mitwindfarm import Layout

from WES2024 import utils

LES_BASE = Path("/scratch/08445/tg877441/control_WES_R1")  # only needed to re-read the LES (generate_les_* --les)
TILED_DIR = utils.CACHEDIR / "generate_les_setpoints_tiled5x5" / "tiled"

NX = NY = 5  # rows in a column, columns
SPACING = 6.0  # [D], both directions

CASES = {
    "NoControl": dict(control="NoControl", model="TANDEM", les="square_NoControl", csv="tandemwall_bem_NoControl_w000.0"),
    "Thrust_TANDEM": dict(control="Thrust", model="TANDEM", les="tandemwall_bem_Thrust_w000.0", csv="tandemwall_bem_Thrust_w000.0"),
    "Yaw_TANDEM": dict(control="Yaw", model="TANDEM", les="tandemwall_bem_YawKOmega_w000.0",
                       csv="tandemwall_bem_YawKOmega_w000.0"),
    "YawMod_TANDEM": dict(control="YawMod", model="TANDEM", les="tandemwall_bem_YawMod_w000.0", csv="tandemwall_bem_YawMod_w000.0"),
    "Joint_TANDEM": dict(control="Joint", model="TANDEM", les="tandemwall_bem_Joint_w000.0", csv="tandemwall_bem_Joint_w000.0"),
    "Joint_Vortex": dict(control="Joint", model="Vortex", les="vortex_bem_Joint_w000.0", csv="vortex_bem_Joint_w000.0"),
    "Joint_Gauss": dict(control="Joint", model="Gauss", les="gaussBP_bem_Joint_w000.0", csv="gaussBP_bem_Joint_w000.0"),
}

# each wake model's own NoControl optimiser CSV (its predicted No-control Cp: the normalisation of its Joint prediction)
NOCONTROL_CSV = {"TANDEM": "tandemwall_bem_NoControl_w000.0", "Vortex": "vortex_bem_NoControl_w000.0",
                 "Gauss": "gaussBP_bem_NoControl_w000.0"}


def column_layout(wdir=0.0) -> Layout:
    return Layout(SPACING * np.arange(NX), np.zeros(NX)).rotate(wdir)


def farm_layout(wdir=0.0) -> Layout:
    # turbine index = column * NX + position in column, so a tiled 5 x 1 solution is np.tile(x, NY) per variable
    xs, ys = np.meshgrid(SPACING * np.arange(NX), SPACING * np.arange(NY), indexing="xy")
    return Layout(xs.ravel(), ys.ravel()).rotate(wdir)


def les_path(key: str) -> Path:
    return LES_BASE / CASES[key]["les"]


def load_setpoints_column0(csv_stem: str) -> pd.DataFrame:
    """One 5-row (row = 0..4, column 0) slice of a tiled optimiser CSV, sorted by row. Columns used downstream: pitch, tsr, yaw (the BEM
    set points, rad), pitch_deg, yaw_deg, Ctprime, Cp (model Cp of that turbine in the 5x5 farm), Cp_farm."""
    df = pd.read_csv(TILED_DIR / f"{csv_stem}.csv")
    return df[(df["column"] == 0) & (df["row"] < 5)].sort_values("row").reset_index(drop=True)


def case_setpoints(key: str) -> pd.DataFrame:
    return load_setpoints_column0(CASES[key]["csv"])


def model_nocontrol_setpoints(model: str) -> pd.DataFrame:
    return load_setpoints_column0(NOCONTROL_CSV[model])
