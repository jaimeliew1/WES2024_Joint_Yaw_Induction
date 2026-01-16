# import dualitic
from pathlib import Path

import numpy as np
from MITRotor.ReferenceTurbines import IEA15MW
from MITRotor.Aerodynamics import DefaultAerodynamics

from mitwindfarm import Layout, Area
from gch.Windfarm import Windfarm

from gch.Superposition import FreestreamQuadratic
from gch.BEM_gradients import DualBEM
from gch.Rotor import BEMUnifiedMomentumLUT, BEM, UnifiedLUTAD
from gch.optimize import JointControl, JointControlBEM
from gch.RotorGrid import FlorisGrid
from gch.Wake import GaussCurlHybridModel

FILESTEM = Path(__file__).stem


PARALLEL = True

windfarm_AD = Windfarm(rotor_model=UnifiedLUTAD(
    rotor_grid=FlorisGrid(),
),
    wake_model=GaussCurlHybridModel(
    rotor_grid=FlorisGrid()
    ))



windfarm_BEM = Windfarm(
    superposition=FreestreamQuadratic(),
        rotor_model=BEM(IEA15MW(),
                    BEM_model=DualBEM,
                    momentum_model=BEMUnifiedMomentumLUT(averaging="rotor_induction_tiploss"),
                    wake_velocities_grid=FlorisGrid(5),
                    aerodynamic_model = DefaultAerodynamics()),
    wake_model=GaussCurlHybridModel(
        rotor_grid=FlorisGrid(5))
        )

layout = Layout(np.array([0.0]), np.array([0.0]), np.array([80/90.0]))


if __name__ == "__main__":
    sol_AD = JointControl(layout, windfarm_AD).optimise(
        use_gradients=False, 
        verbose=False)
    sol_BEM = JointControlBEM(layout, windfarm_BEM).optimise(use_gradients=False, verbose=False)

    # sol_BEM = windfarm_BEM(layout, [(0, 8, 0)])
    

    print("~~~ Optimized AD Solution ~~~")
    rotor_sol = sol_AD.rotors[0]
    print(f"{rotor_sol.Ctprime=}")
    print(f"{rotor_sol.Cp=}")

    print()
    print("~~~ Optimized BEM Solution ~~~")
    rotor_sol = sol_BEM.rotors[0]
    print(f"{rotor_sol.extra.pitch=}")
    print(f"{rotor_sol.extra.tsr=}")
    print(f"{rotor_sol.Cp=}")
    print(f"{rotor_sol.Ct=}")

    print()
