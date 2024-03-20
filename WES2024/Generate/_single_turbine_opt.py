from pathlib import Path

import numpy as np
from MITRotor.ReferenceTurbines import IEA15MW
from mitwindfarm.Layout import Layout
from mitwindfarm.Rotor import BEM
from mitwindfarm.windfarm import Windfarm

from WES2024.BEM_gradients import DualBEM
from WES2024.optimise import JointControlBEM

FILESTEM = Path(__file__).stem


PARALLEL = True


windfarm = Windfarm(rotor_model=BEM(IEA15MW(), BEM_model=DualBEM))
layout = Layout(np.array([0.0]), np.array([0.0]))


if __name__ == "__main__":
    sol = JointControlBEM(layout, windfarm).optimise(use_gradients=True)
    rotor_sol = sol.rotors[0]
    print(f"{rotor_sol.extra.pitch=}")
    print(f"{rotor_sol.extra.tsr=}")
    print(f"{rotor_sol.Cp=}")
