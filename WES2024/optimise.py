import numpy as np
from mitwindfarm.windfarm import WindfarmSolution
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from dualitic import DualVariables


class Controller:
    def __init__(self, layout, windfarm):
        self.layout = layout
        self.windfarm = windfarm
        self.N = len(layout)

    def optimise(self, verbose=False, use_gradients=True) -> WindfarmSolution:
        x0 = self.initial_guess()
        bounds = self.bounds()

        if use_gradients:
            sol = minimize(self.grad_objective_func, x0, bounds=bounds, jac=True)
        else:
            sol = minimize(self.objective_func, x0, bounds=bounds)
        if verbose:
            print(f"{sol.nfev=}")
            print(f"{sol.njev=}")
        asdf = self.solve_for_setpoints(sol.x)
        return asdf

    def objective_func(self, x):
        windfarm_sol = self.solve_for_setpoints(x)
        return -windfarm_sol.Cp()

    def grad_objective_func(self, x):
        x = DualVariables(x)
        windfarm_sol = self.solve_for_setpoints(x)
        Cp = windfarm_sol.Cp().real[0]
        Cp_grad = windfarm_sol.Cp().dual[0]
        return -Cp, -Cp_grad


class NoControl(Controller):
    def __init__(self, layout, windfarm):
        self.layout = layout
        self.windfarm = windfarm
        self.N = len(layout)

    def optimise(self, **kwargs) -> WindfarmSolution:
        setpoints = [(2.0, 0.0) for _ in range(self.N)]
        return self.windfarm(self.layout, setpoints)


class YawControl(Controller):
    def initial_guess(self) -> ArrayLike:
        return [0.0001 for _ in range(self.N)]

    def bounds(self) -> list:
        return [tuple(np.deg2rad((-50, 50)))]

    def solve_for_setpoints(self, x) -> WindfarmSolution:
        setpoints = [(2.0, _x) for _x in x]
        return self.windfarm(self.layout, setpoints)


class ThrustControl(Controller):
    def initial_guess(self) -> ArrayLike:
        return [2.0 for _ in range(self.N)]

    def bounds(self) -> list:
        return [(0.00001, 3.0) for _ in range(self.N)]

    def solve_for_setpoints(self, x) -> WindfarmSolution:
        setpoints = list((_x, 0.0) for _x in x)
        return self.windfarm(self.layout, setpoints)


class JointControl(Controller):
    def initial_guess(self) -> ArrayLike:
        return [2.0 for _ in range(self.N)] + [0.0 for _ in range(self.N)]

    def bounds(self) -> list:
        return [(0.00001, 2.0) for _ in range(self.N)] + [tuple(np.deg2rad((-50, 50))) for _ in range(self.N)]

    def solve_for_setpoints(self, x) -> WindfarmSolution:
        setpoints = list((_x1, _x2) for _x1, _x2 in zip(x[: self.N], x[self.N :]))
        return self.windfarm(self.layout, setpoints)
