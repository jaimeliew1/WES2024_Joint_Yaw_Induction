import numpy as np
from UnifiedMomentumModel.Momentum import Heck, MomentumSolution
from mitwindfarm.Rotor import Rotor, Area, RotorGrid
from mitwindfarm import Windfield, RotorSolution
# from WES2024.LES_new.UpfalCustom.SetpointCurve import SetpointCurve

class AD(Rotor):
    """
    Axial Distribution rotor model.

    Methods:
    - __call__(Ctprime, yaw): Calculate the rotor solution for given Ctprime and yaw inputs.
    """

    def __init__(self, rotor_grid: RotorGrid = None):
        """
        Initialize the AD rotor model using the Heck momentum model.
        """
        self._model = Heck()
        if rotor_grid is None:
            self.rotor_grid = Area()
        else:
            self.rotor_grid = rotor_grid

    def __call__(self, x: float, y: float, z: float, windfield: Windfield, Ctprime, yaw) -> RotorSolution:
        """
        Calculate the rotor solution for given Ctprime and yaw inputs.

        Parameters:
        - Ctprime (float): Thrust coefficient including the effect of yaw.
        - yaw (float): Yaw angle of the rotor.

        Returns:
        RotorSolution: The calculated rotor solution.
        """
        # Calculate rotor solution (independent of wind field in this model)
        sol: MomentumSolution = self._model(Ctprime, yaw)

        # Get the points over rotor to be sampled in windfield
        xs_loc, ys_loc, zs_loc = self.rotor_grid.grid_points()
        xs_glob, ys_glob, zs_glob = xs_loc + x, ys_loc + y, zs_loc + z

        # sample windfield and calculate rotor effective wind speed
        Us = windfield.wsp(xs_glob, ys_glob, zs_glob)
        
        x = x * np.array([1])
        y = y * np.array([1])
        z = z * np.array([1])

        REWS = self.rotor_grid.average(Us)
        RETI = windfield.RETI(x, y, z)

        # rotor solution is normalised by REWS. Convert normalisation to U_inf and return
        return RotorSolution(
            yaw,
            sol.Cp * REWS**3,
            sol.Ct * REWS**2,
            sol.Ctprime,
            sol.an * REWS,
            sol.u4 * REWS,
            sol.v4 * REWS,
            REWS,
            TI=RETI,
            extra=sol,
        )


class AnalyticalAD(Rotor):
    """
    Actuator disk rotor model using analytically line averaged REWS.

    Methods:
    - __call__(Ctprime, yaw): Calculate the rotor solution for given Ctprime and yaw inputs.
    """

    def __init__(self):
        """
        Initialize the AD rotor model using the Heck momentum model.
        """
        self._model = Heck()

    def __call__(
        self, x: float, y: float, z: float, windfield: Windfield, Ctprime, yaw
    ) -> RotorSolution:
        """
        Calculate the rotor solution for given Ctprime and yaw inputs.

        Parameters:
        - Ctprime (float): Thrust coefficient including the effect of yaw.
        - yaw (float): Yaw angle of the rotor.

        Returns:
        RotorSolution: The calculated rotor solution.
        """
        # Calculate rotor solution (independent of wind field in this model)
        sol: MomentumSolution = self._model(Ctprime, yaw)

        x = np.array([1]) * x
        y = np.array([1]) * y
        z = np.array([1]) * z

        # analytically line-averaged rotor wind speed and turbulence intensity
        REWS = windfield.line_wsp(x, y, z)
        RETI = windfield.RETI(x, y, z)

        # rotor solution is normalised by REWS. Convert normalisation to U_inf and return
        return RotorSolution(
            yaw,
            sol.Cp * REWS**3,
            sol.Ct * REWS**2,
            sol.Ctprime,
            sol.an * REWS,
            sol.u4 * REWS,
            sol.v4 * REWS,
            REWS,
            TI=RETI,
            extra=sol,
        )
    

# class FixedControlAnalyticalAD(Rotor):
#     """
#     Actuator disk rotor model with a fixed thrust setpoint control strategy
#     based on wind speed at rotor using analytically line averaged REWS.

#     Methods:
#     - __call__(x, y, z, windfield): Calculate the rotor solution for given Ctprime and yaw inputs.
#     """

#     def __init__(
#         self
#     ):
#         """
#         Initialize the AD rotor model using the Heck momentum model.
#         """
#         self._model = Heck()
#         self.setpoint_curve = SetpointCurve()

#     def __call__(
#         self, x: float, y: float, z: float, windfield: Windfield, u_rated: float
#     ) -> RotorSolution:
#         """
#         Calculate the rotor solution for given Ctprime and yaw inputs. If
#         Ctprime is not given, use setpoint based on ThrustCurve, if yaw is
#         not given, assume yaw is zero.

#         Parameters:
#         - x (float): the x coordinate of the rotor.
#         - y (float): the y coordinate of the rotor.
#         - z (float): the z coordinate of the rotor.
#         - windfield (Windfield): the windfield in which the rotor is operating.
#         - u_rated (float): the rated wind speed of the turbine
#                 normalized by the free stream wind speed.

#         Returns:
#         RotorSolution: The calculated rotor solution.
#         """

#         x = np.array([1]) * x
#         y = np.array([1]) * y
#         z = np.array([1]) * z

#         # analytically line-averaged rotor wind speed and turbulence intensity
#         REWS = windfield.line_wsp(x, y, z)
#         RETI = windfield.RETI(x, y, z)

#         # get Ctprime from SetpointCurve
#         Ctprime = self.setpoint_curve(REWS / u_rated)
#         yaw = 0.0
        
#         # Calculate rotor solution (independent of wind field in this model)
#         sol: MomentumSolution = self._model(Ctprime, yaw)

#         # rotor solution is normalised by REWS. Convert normalisation to U_inf and return
#         return RotorSolution(
#             yaw,
#             sol.Cp * REWS**3,
#             sol.Ct * REWS**2,
#             sol.Ctprime,
#             sol.an * REWS,
#             sol.u4 * REWS,
#             sol.v4 * REWS,
#             REWS,
#             TI=RETI,
#             extra=sol,
#         )
