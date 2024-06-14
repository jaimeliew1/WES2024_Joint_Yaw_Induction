import dualitic
from pathlib import Path

import numpy as np
from MITRotor.ReferenceTurbines import IEA15MW
from mitwindfarm import BEM, Layout, Windfarm

from WES2024.BEM_gradients import DualBEM
from WES2024.CustomRotors import BEMUnifiedMomentumLUT, UnifiedLUTAD
from WES2024.optimise import JointControlBEM, JointControl

FILESTEM = Path(__file__).stem


PARALLEL = True

windfarm_AD = Windfarm(rotor_model=UnifiedLUTAD())


windfarm_BEM = Windfarm(
    rotor_model=BEM(IEA15MW(), BEM_model=DualBEM, momentum_model=BEMUnifiedMomentumLUT())
)
layout = Layout(np.array([0.0]), np.array([0.0]))


if __name__ == "__main__":

    sol_AD = JointControl(layout, windfarm_AD).optimise(use_gradients=True, verbose=False)
    sol_BEM = JointControlBEM(layout, windfarm_BEM).optimise(use_gradients=True, verbose=False)

    rotor_sol = sol_AD.rotors[0]
    print(f"{rotor_sol.Ctprime=}")
    print(f"{rotor_sol.Cp=}")

    print()

    rotor_sol = sol_BEM.rotors[0]
    print(f"{rotor_sol.extra.pitch=}")
    print(f"{rotor_sol.extra.tsr=}")
    print(f"{rotor_sol.Cp=}")
