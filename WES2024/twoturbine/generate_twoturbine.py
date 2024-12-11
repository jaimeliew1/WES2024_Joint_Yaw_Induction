"""
Generate two-turbine data usign MITWindfarm. 

This will take a bit of debugging, probably. 

Kirby Heck
2024 Nov 12
"""

from mitwindfarm import GridLayout, Windfarm
from mitwindfarm.Wake import GaussianWakeModel
from mitwindfarm.Rotor import AD
from WES2024.CustomRotors import UnifiedLUTAD, CosineAD
from WES2024.LES.run import TIAMB  # = 0.053
from WES2024.LES.run_old import VariableKwGaussianWakeModel

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from itertools import product
from tqdm import tqdm
from pathlib import Path

# DATAPATH = Path(__file__).parent
DATAPATH = Path(r"C:\MIT\howland\python_scripts\WES2024_Joint_Yaw_Induction_Final_analysis\data")

rotor_lookup = dict(unified=UnifiedLUTAD(), cosine=CosineAD(Pp=1.9), jfm=AD())

def run(rotormodel="unified"):
    print("Generating sweep for rotor:", rotormodel)
    layout = GridLayout(6.0133, 0.0, 2, 1).rotate(3.8)
    kw_params = (0.0, 0.91948314, -0.00896332)  # from calibration
    # kw_params = (0, 0, 0.0398)

    # ugh try a few different things here... 
    wf = Windfarm(
        rotor_model=rotor_lookup[rotormodel], wake_model=GaussianWakeModel(sigma=1/np.sqrt(8), kw=0.04, x0=3, ), 
        # rotor_model=rotor_lookup[rotormodel], wake_model=GaussianWakeModel(sigma=0.25, kw=0.07), 
        # rotor_model=rotor_lookup[rotormodel], wake_model=VariableKwGaussianWakeModel(*kw_params), TIamb=TIAMB
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
    run("jfm")
