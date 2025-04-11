from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike

import MITRotor
from MITRotor.Aerodynamics import AerodynamicProperties
from MITRotor.Geometry import BEMGeometry
from MITRotor.RotorDefinition import RotorDefinition


class NoTiplossTangentialInduction(MITRotor.TangentialInductionModel):
    def __call__(
        self,
        aero_props: AerodynamicProperties,
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: RotorDefinition,
        geom: BEMGeometry,
    ) -> ArrayLike:
        tangential_integral = aero_props.W**2 * aero_props.C_tan

        aprime = (
            aero_props.solidity
            / (4 * np.maximum(geom.mu_mesh, 0.1) ** 2 * tsr * (1 - aero_props.an) * np.cos(yaw))
            * tangential_integral
        )

        return aprime