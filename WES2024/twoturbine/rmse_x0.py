"""
Generate two-turbine data to compare with LES.

Kirby Heck
2024 Nov 12
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from itertools import product
from tqdm import tqdm
from pathlib import Path

from mitwindfarm import GridLayout, Windfarm
from mitwindfarm.Rotor import AD
from WES2024.CustomRotors import UnifiedLUTAD, CosineAD
from WES2024.LES.shared import TIAMB
from WES2024.LES_new.final_calibration import VariableKwGaussianWakeModel, get_wakemodel

# other wake model
x0_3_wm = VariableKwGaussianWakeModel(a=0.7683, b=0, c=0.004825, sigma=1 / np.sqrt(8), x0=3)
x0_1_wm = get_wakemodel()


DATAPATH = Path(__file__).parent

rotor_lookup = dict(
    unified=UnifiedLUTAD(), cosine=CosineAD(Pp=1.9), cosine3=CosineAD(Pp=3), jfm=AD()
)


def run_wf(
    wakemodel,
    ctp_list,
    yaw_list,
    rotormodel="unified",
):
    print("Generating sweep for rotor:", rotormodel)
    layout = GridLayout(6.0133, 0.0, 2, 1).rotate(3.8)

    # ugh try a few different things here...
    wf = Windfarm(rotor_model=rotor_lookup[rotormodel], wake_model=wakemodel, TIamb=TIAMB)

    # now we need to sweep over all the set points
    # I don't think this is vectorized
    sp2 = (2, 0)  # waked turbine is always at Betz
    solutions = []
    for sp1 in tqdm(zip(ctp_list, yaw_list)):
        sol = wf(layout, [sp1, sp2])
        solutions.append(sol)

    # now make this into a dataframe
    df = pl.DataFrame(
        {
            "ctp": [sol.rotors[0].Ctprime for sol in solutions],
            "yaw": np.rad2deg([sol.rotors[0].yaw for sol in solutions]),
            "Cp_1": [sol.rotors[0].Cp for sol in solutions],
            "Cp_2": [sol.rotors[1].Cp for sol in solutions],
            "Cp_T": [sol.Cp for sol in solutions],
        }
    )

    return df


def get_les():
    return pl.read_csv(DATAPATH / "twoturbine_alldata_LES.csv")


def run():
    """Compares predictions for different far wake models"""
    df_les = get_les().with_columns(pl.lit(0).alias("x0"))
    df_x01 = run_wf(x0_1_wm, df_les['ctp'].to_numpy(), np.deg2rad(df_les['yaw'].to_numpy())).with_columns(pl.lit(1).alias("x0"))
    df_x03 = run_wf(x0_3_wm, df_les['ctp'].to_numpy(), np.deg2rad(df_les['yaw'].to_numpy())).with_columns(pl.lit(3).alias("x0"))

    # compute RMSE
    cols = ['Cp_1', "Cp_2", "Cp_T"]
    err_01 = np.abs(df_x01.select(cols) - df_les.select(cols)) / df_les.select(cols)
    err_03 = np.abs(df_x03.select(cols) - df_les.select(cols)) / df_les.select(cols)

    
    print("RMSE for x0=1: ", err_01.mean(axis=0))
    print("RMSE for x0=3: ", err_03.mean(axis=0))
    
    print("done")


if __name__ == "__main__":
    run()
