import dualitic
from pathlib import Path

import numpy as np
from MITRotor.ReferenceTurbines import IEA15MW
from mitwindfarm import Layout, Area
from gch.Windfarm import Windfarm
from gch.Superposition import FreestreamQuadratic
from gch.BEM_gradients import DualBEM
from gch.Rotor import BEMUnifiedMomentumLUT, BEM, UnifiedLUTAD
from gch.optimize import JointControl, JointControlBEM
from gch.RotorGrid import FlorisGrid
from gch.Wake import GaussCurlHybridModel
import matplotlib.pyplot as plt
from WES2024 import utils

FILESTEM = Path(__file__).stem

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
                    momentum_model=BEMUnifiedMomentumLUT(averaging="rotor"),
                    wake_velocities_grid=FlorisGrid(5)),
    wake_model=GaussCurlHybridModel(
        rotor_grid=FlorisGrid(5))
        )

layout = Layout(np.array([0.0]), np.array([0.0]), np.array([80/90.0]))


def _solve_AD_for_setpoints(ctprime):
    sol_AD = windfarm_AD(layout, [(ctprime, 0)])
    return sol_AD.rotors[0].Cp

# We want to test the optimizer JointControl
def test_AD_optimizer():
    # first do a brute-force sweep over ctprime
    Ctprimes = np.linspace(0.1, 3.0, 20)
    Cps = [_solve_AD_for_setpoints(ctp) for ctp in Ctprimes]
    
    sol_opt = JointControl(layout, windfarm_AD).optimise()
    
    fig, ax = plt.subplots()
    ax.plot(Ctprimes, Cps)

    # brute-force optimum
    opt_idx = np.argmax(Cps)
    ax.plot(Ctprimes[opt_idx], Cps[opt_idx], 'o', label='Brute-force Optimum', markersize=10)

    # optimized point
    ax.plot(sol_opt.rotors[0].Ctprime, sol_opt.rotors[0].Cp, '*', label='Optimized Point', markersize=10)


    ax.set_xlabel("Ctprime")
    ax.set_ylabel("Cp")
    ax.legend()
    plt.savefig(utils.FIGDIR / f"{FILESTEM}_Cp_vs_Ctprime.png")

# same for JointControlBEM
def _solve_BEM_for_setpoints(tsr):
    sol_BEM = windfarm_BEM(layout, [(0, tsr, 0)])
    return sol_BEM.rotors[0].Cp

# We want to test the optimizer JointControlBEM
def test_BEM_optimizer():
    # first do a brute-force sweep over ctprime
    TSRs = np.linspace(6, 12, 20)
    Cps = [_solve_BEM_for_setpoints(tsr) for tsr in TSRs]
    
    sol_opt = JointControlBEM(layout, windfarm_BEM).optimise()
    
    fig, ax = plt.subplots()
    ax.plot(TSRs, Cps)

    # brute-force optimum
    opt_idx = np.argmax(Cps)
    ax.plot(TSRs[opt_idx], Cps[opt_idx], 'o', label='Brute-force Optimum', markersize=10)

    # optimized point
    ax.plot(sol_opt.rotors[0].tsr, sol_opt.rotors[0].Cp, '*', label='Optimized Point', markersize=10)


    ax.set_xlabel("TSR")
    ax.set_ylabel("Cp")
    ax.legend()
    plt.savefig(utils.FIGDIR / f"{FILESTEM}_BEM.png")



if __name__ == "__main__":
    # test_AD_optimizer()
    test_BEM_optimizer()

