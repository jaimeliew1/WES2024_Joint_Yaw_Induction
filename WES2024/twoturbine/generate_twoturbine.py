"""
Generate two-turbine data usign MITWindfarm. 

This will take a bit of debugging, probably. 

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
from WES2024.LES_new.final_calibration import (
    VariableKwGaussianWakeModel,
    get_calibration_params
)

DATAPATH = Path(__file__).parent

rotor_lookup = dict(unified=UnifiedLUTAD(), cosine=CosineAD(Pp=1.9), cosine3=CosineAD(Pp=3), jfm=AD())

def run(rotormodel="unified"):
    print("Generating sweep for rotor:", rotormodel)
    layout = GridLayout(6.0133, 0.0, 2, 1).rotate(3.8)
    kw_params = get_calibration_params()
    print("Using calibration parameters a*TI + b*Ctprime + c: (a, b, c) =", kw_params)

    # ugh try a few different things here... 
    wf = Windfarm(
        rotor_model=rotor_lookup[rotormodel], wake_model=VariableKwGaussianWakeModel(**kw_params), TIamb=TIAMB
    )

    # now we need to sweep over all the set points
    # I don't think this is vectorized
    sp2 = (2, 0)  # waked turbine is always at Betz
    solutions = []
    yaws = np.deg2rad(np.linspace(0, 45))
    ctps = np.arange(0.4, 4.5, 0.1)
    for sp1 in tqdm(product(ctps, yaws)): 
        sol = wf(layout, [sp1, sp2])
        solutions.append(sol)

    # now make this into a dataframe
    df = pl.DataFrame({
        'ctp': [sol.rotors[0].Ctprime for sol in solutions], 
        'yaw': np.rad2deg([sol.rotors[0].yaw for sol in solutions]), 
        'Cp_1': [sol.rotors[0].Cp for sol in solutions], 
        'Cp_2': [sol.rotors[1].Cp for sol in solutions], 
        'Cp_T': [sol.Cp for sol in solutions], 
    })

    df.write_csv(DATAPATH / f"twoturbine_mitwindfarm_{rotormodel}.csv")


if __name__ == "__main__":
    run("unified")
    run("cosine")
    run("cosine3")
    run("jfm")
