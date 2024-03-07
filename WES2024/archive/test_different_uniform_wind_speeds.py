from pathlib import Path

import numpy as np

from mitwindfarm.Rotor import BEM
from mitwindfarm.Windfield import Uniform
from MITRotor.ReferenceTurbines import IEA15MW
from rich import print

figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    rotor = IEA15MW()
    bem = BEM(rotor)

    Us = np.round(np.arange(0.1, 1.01, 0.1), 2)
    # yaws = np.linspace(-30, 30)
    sols = []
    for U in Us:
        windfield = Uniform(U)
        sol = bem(0.0, 0.0, 0.0, windfield, 0, 7, 0)
        sols.append(sol)
    

    # renormalised a_n
    for U, sol in zip(Us, sols):
        print(
            U,
            sol,
            f"{sol.an=:2.3f}",
            f"{sol.Ct=:2.3f}",
            f"{sol.Ctprime=:2.3f}",
        )
        # print(U, f"{a=}")
