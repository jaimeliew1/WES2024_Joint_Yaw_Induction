import matplotlib.pyplot as plt
import numpy as np
from MITRotor.Geometry import BEMGeometry
from MITRotor.ReferenceTurbines import IEA15MW
from mitwindfarm.Windfield import PowerLaw
from rich import print
from WES2024.BEM_gradients import DualBEM
from dualitic import DualNumber

PITCH, TSR, YAW = np.deg2rad(0), 7, np.deg2rad(0)
zhub = DualNumber(0.7, [1])  # hub height in rotor diameters
geom = BEMGeometry(3, 10)
if __name__ == "__main__":
    windfield = PowerLaw(1.0, zhub, 0.12)
    bem = DualBEM(IEA15MW(), geom)
    X, Y, Z = bem.bem.sample_points()
    X /= 2
    Y /= 2
    Z = Z / 2 + zhub

    U = windfield.wsp(X, Y, Z)
    # print(U)
    sol = bem(PITCH, TSR, DualNumber(YAW, [0]), U)
    print(sol)
    print(sol.Cp())
