from abc import ABC, abstractmethod
import numpy as np
from mitwindfarm import WindfarmSolution
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from dualitic import DualVariables


class Controller(ABC):
    """Abstract base class for wind farm control optimisation objects."""

    @abstractmethod
    def initial_guess(self) -> ArrayLike:
        """Return an initial guess for optimization."""
        pass

    @abstractmethod
    def bounds(self) -> ArrayLike:
        """Return the bounds for optimization."""
        pass

    @abstractmethod
    def solve_for_setpoints(self, x) -> WindfarmSolution:
        """Solve for setpoints based on the input variables.

        Parameters:
            x: Input variables.

        Returns:
            WindfarmSolution: Solution for the wind farm.
        """
        pass

    def __init__(self, layout, windfarm, Komega_constraint=False):
        self.layout = layout
        self.windfarm = windfarm
        self.N = len(layout)
        self.Komega_constraint = Komega_constraint

    def optimise(self, verbose=False, use_gradients=True, Cp_constraint=None) -> WindfarmSolution:
        """Optimize the wind farm layout.

        Parameters:
            verbose (bool): Whether to print optimisation information.
            use_gradients (bool): Whether to use gradients in optimization.
            Cp_constraint (float): Maximum Cp constraint for all turbines. None = no constraint.

        Returns:
            WindfarmSolution: Optimized wind farm solution.
        """
        x0 = self.initial_guess()
        bounds = self.bounds()

        if Cp_constraint:
            self._Cp_max = Cp_constraint
            constraint = dict(type="ineq", fun=self.constraint_func)
            if use_gradients:
                constraint["jac"] = self.constraint_jac_func
        else:
            constraint = None
        if self.Komega_constraint:
            constraint = dict(type="eq", fun=self.K_omega_constraint_func)
            if use_gradients:
                constraint["jac"] = self.K_omega_constraint_jac_func
        else:
            constraint = None

        if use_gradients:
            sol = minimize(
                self.grad_objective_func, x0, bounds=bounds, jac=True, constraints=constraint
            )
        else:
            sol = minimize(self.objective_func, x0, bounds=bounds, constraints=constraint)

        if verbose:
            print(f"{sol.nfev=}")
            print(f"{sol.njev=}")
            print(sol)
        optimized_solution = self.solve_for_setpoints(sol.x)
        return optimized_solution

    def objective_func(self, x) -> float:
        """Calculate the objective function.

        Parameters:
            x: Input variables.

        Returns:
            float: Objective function value.
        """
        windfarm_sol = self.solve_for_setpoints(x)
        return -windfarm_sol.Cp

    def grad_objective_func(self, x) -> tuple[float, ArrayLike]:
        """Calculate the combined objective function and its gradient.

        Parameters:
            x: Input variables.

        Returns:
            Tuple[float, ArrayLike]: Objective function value and its gradient.
        """
        x = DualVariables(x)
        windfarm_sol = self.solve_for_setpoints(x)
        Cp = windfarm_sol.Cp.real[0]
        Cp_grad = windfarm_sol.Cp.dual[0]
        return -Cp, -Cp_grad

    def constraint_func(self, x) -> list[float]:
        """Calculate the Cp constraint function on all turbines.

        Parameters:
            x: Input variables.

        Returns:
            List[float]: List of constraint values.
        """
        x = DualVariables(x)
        sol = self.solve_for_setpoints(x)
        Cps = [self._Cp_max - x.Cp.real[0] for x in sol.rotors]
        self._grad = [-x.Cp.dual[0] for x in sol.rotors]
        return Cps

    def constraint_jac_func(self, x) -> ArrayLike:
        """
        Returns jacobian of constraint function. retrieved from cached call of
        constraint_func.

        Parameters:
            x: Input variables.

        Returns:
            List[float]: Jacobian of the constraint function.
        """
        return self._grad
# Calculated in separate optimisation (see WES2024.Generate._single_turbine_opt)
CTPRIME_OPT = 2.10418397932219
CTPRIME_OPT = 2.0
class NoControl(Controller):
    def optimise(self, **kwargs) -> WindfarmSolution:
        setpoints = [(CTPRIME_OPT, 0.0) for _ in range(self.N)]
        return self.windfarm(self.layout, setpoints)

    def initial_guess(self):
        ...

    def bounds(self):
        ...

    def solve_for_setpoints(self, x):
        ...


class YawControl(Controller):
    def initial_guess(self) -> ArrayLike:
        return [0.0001 for _ in range(self.N)]

    def bounds(self) -> list:
        return [tuple(np.deg2rad((-50, 50)))]

    def solve_for_setpoints(self, x) -> WindfarmSolution:
        setpoints = [(CTPRIME_OPT, _x) for _x in x]
        return self.windfarm(self.layout, setpoints)


class ThrustControl(Controller):
    def initial_guess(self) -> ArrayLike:
        return [0.2 for _ in range(self.N)]

    def bounds(self) -> list:
        return [(0.00001, 4.0) for _ in range(self.N)]

    def solve_for_setpoints(self, x) -> WindfarmSolution:
        setpoints = list((_x, 0.0) for _x in x)
        return self.windfarm(self.layout, setpoints)


class JointControl(Controller):
    def initial_guess(self) -> ArrayLike:
        return [0.2 for _ in range(self.N)] + [0.0 for _ in range(self.N)]

    def bounds(self) -> list:
        return [(0.00001, 4.0) for _ in range(self.N)] + [
            tuple(np.deg2rad((-50, 50))) for _ in range(self.N)
        ]

    def solve_for_setpoints(self, x) -> WindfarmSolution:
        setpoints = list((_x1, _x2) for _x1, _x2 in zip(x[: self.N], x[self.N :]))
        return self.windfarm(self.layout, setpoints)


# Calculated in separate optimisation (see WES2024.Generate._single_turbine_opt)

# Setpoints using Heck momentum model with high thrust correction (DEPRECIATED)
# PITCH_OPT = -0.018804860075115795
# TSR_OPT = 9.138197665010335
# CP_OPT = 0.5065639542511471

# Setpoints using unified momentum model
PITCH_OPT = -0.023146163628916267
TSR_OPT = 9.23061314763139
CP_OPT = 0.5072138998869634


class NoControlBEM(Controller):
    def optimise(self, **kwargs) -> WindfarmSolution:
        setpoints = [(PITCH_OPT, TSR_OPT, 0.0) for _ in range(self.N)]
        return self.windfarm(self.layout, setpoints)

    def initial_guess(self):
        ...

    def bounds(self):
        ...

    def solve_for_setpoints(self, x):
        ...


class ThrustControlBEM(Controller):
    def initial_guess(self) -> ArrayLike:
        return [PITCH_OPT for _ in range(self.N)] + [TSR_OPT for _ in range(self.N)]

    def bounds(self) -> list:
        return [(-np.deg2rad(10), np.deg2rad(10)) for _ in range(self.N)] + [
            (3, 10) for _ in range(self.N)
        ]

    def solve_for_setpoints(self, x) -> WindfarmSolution:
        setpoints = list((_x1, _x2, 0.0) for _x1, _x2 in zip(x[: self.N], x[self.N :]))
        return self.windfarm(self.layout, setpoints)


class YawControlBEM(Controller):
    def initial_guess(self) -> ArrayLike:
        return [0.0 for _ in range(self.N)]

    def bounds(self) -> list:
        return [(-np.deg2rad(45), np.deg2rad(45)) for _ in range(self.N)]

    def solve_for_setpoints(self, x) -> WindfarmSolution:
        setpoints = list((PITCH_OPT, TSR_OPT, _x3) for _x3 in x)
        return self.windfarm(self.layout, setpoints)


class YawControlKOmegaBEM(Controller):
    def __init__(self, *args, Komega_constraint=True, **kwargs):
        super().__init__(*args, Komega_constraint=Komega_constraint, **kwargs)

    def initial_guess(self) -> ArrayLike:
        return [TSR_OPT for _ in range(self.N)] + [0.0 for _ in range(self.N)]

    def bounds(self) -> list:
        return [(1, 10) for _ in range(self.N)] + [
            (-np.deg2rad(45), np.deg2rad(45)) for _ in range(self.N)
        ]

    def solve_for_setpoints(self, x) -> WindfarmSolution:
        setpoints = list((PITCH_OPT, _x2, _x3) for _x2, _x3 in zip(x[: self.N], x[self.N :]))
        return self.windfarm(self.layout, setpoints)

    def K_omega_constraint_func(self, x) -> list[float]:
        """Calculate the constraint function on all turbines.

        Parameters:
            x: Input variables.

        Returns:
            List[float]: List of constraint values.
        """
        x = DualVariables(x)
        sol = self.solve_for_setpoints(x)
        constraints_dual = [
            rotor.Cp / rotor.extra.tsr**3 - CP_OPT / TSR_OPT**3 for rotor in sol.rotors
        ]

        constraints = [x.real[0] for x in constraints_dual]
        self._Komega_grad = [x.dual[0] for x in constraints_dual]

        return constraints

    def K_omega_constraint_jac_func(self, x) -> ArrayLike:
        """
        Returns jacobian of constraint function. retrieved from cached call of
        constraint_func.
        """
        return self._Komega_grad


class JointControlBEM(Controller):
    def initial_guess(self) -> ArrayLike:
        return (
            [PITCH_OPT for _ in range(self.N)]
            + [TSR_OPT for _ in range(self.N)]
            + [0.0 for _ in range(self.N)]
        )

    def bounds(self) -> list:
        return (
            [(-np.deg2rad(10), np.deg2rad(10)) for _ in range(self.N)]
            + [(3, 10) for _ in range(self.N)]
            + [(-np.deg2rad(45), np.deg2rad(45)) for _ in range(self.N)]
        )

    def solve_for_setpoints(self, x) -> WindfarmSolution:
        setpoints = list(
            (_x1, _x2, _x3)
            for _x1, _x2, _x3 in zip(x[: self.N], x[self.N : 2 * self.N], x[2 * self.N :])
        )
        return self.windfarm(self.layout, setpoints)
