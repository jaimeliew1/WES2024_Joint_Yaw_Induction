import numpy as np
import polars as pl
from itertools import product
from foreach import foreach

from MITRotor.ReferenceTurbines import IEA15MW
from MITRotor.Aerodynamics import DefaultAerodynamics
from mitwindfarm import BEM, Layout, Windfarm

from WES2024.BEM_gradients import DualBEM
from WES2024.CustomRotors import BEMUnifiedMomentumLUT
from WES2024 import utils

windfarm_BEM = Windfarm(
    rotor_model=BEM(IEA15MW(), 
                    BEM_model=DualBEM, 
                    momentum_model=BEMUnifiedMomentumLUT(averaging='rotor_induction_tiploss'),
                    aerodynamic_model=DefaultAerodynamics(),
                    )
)
layout = Layout(np.array([0.0]), np.array([0.0]))


pitches = np.radians([0, 5, 10])
tsrs = np.linspace(6, 10, 40)
params = product(pitches, tsrs)

def _run_case(x):
    pitch, tsr = x
    sol = windfarm_BEM(layout, [(pitch, tsr, 0)])
    rotor_sol = sol.rotors[0]
    return pl.DataFrame({
        "pitch": [np.degrees(pitch)],
        "tsr": [tsr],
        "Cp": [rotor_sol.Cp],
        "Ct": [rotor_sol.Ct],
    })

@utils.cache_polars(utils.CACHEDIR / 'test_BEM_models_Original_generate.csv')
def generate():
    return pl.concat([_run_case(x) for x in params])

if __name__ == "__main__":
    df = generate(regenerate=True)
    print(df)
