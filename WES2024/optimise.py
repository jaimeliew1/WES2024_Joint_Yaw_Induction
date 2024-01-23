import numpy as np
from mitwindfarm.windfarm import WindfarmSolution
from numpy.typing import ArrayLike
from scipy.optimize import minimize


class NoControl:
    def __init__(self, layout, windfarm):
        self.layout = layout
        self.windfarm = windfarm
        self.N = len(layout)

    def optimise(self) -> WindfarmSolution:
        setpoints = [(2.0, 0.0) for _ in range(self.N)]
        return self.windfarm(self.layout, setpoints)


class Controller:
    def __init__(self, layout, windfarm):
        self.layout = layout
        self.windfarm = windfarm
        self.N = len(layout)

    def optimise(self, verbose=False) -> WindfarmSolution:
        x0 = self.initial_guess()
        bounds = self.bounds()
        sol = minimize(self.objective_func, x0, bounds=bounds)
        if verbose:
            print(f"{sol.nfev=}")
            print(f"{sol.njev=}")
        asdf = self.solve_for_setpoints(sol.x)
        return asdf


class YawControl(Controller):
    def initial_guess(self) -> ArrayLike:
        return [0.0 for _ in range(self.N)]

    def bounds(self) -> list:
        return [tuple(np.deg2rad((-50, 50)))]

    def solve_for_setpoints(self, x) -> WindfarmSolution:
        setpoints = [(2.0, _x) for _x in x]
        return self.windfarm(self.layout, setpoints)

    def objective_func(self, x):
        windfarm_sol = self.solve_for_setpoints(x)
        return -windfarm_sol.Cp()


class ThrustControl(Controller):
    def initial_guess(self) -> ArrayLike:
        return [2.0 for _ in range(self.N)]

    def bounds(self) -> list:
        return [(0.0, 3.0) for _ in range(self.N)]

    def solve_for_setpoints(self, x) -> WindfarmSolution:
        setpoints = list((_x, 0.0) for _x in x)
        return self.windfarm(self.layout, setpoints)

    def objective_func(self, x):
        windfarm_sol = self.solve_for_setpoints(x)
        return -windfarm_sol.Cp()


class JointControl(Controller):
    def initial_guess(self) -> ArrayLike:
        return [2.0 for _ in range(self.N)] + [0.0 for _ in range(self.N)]

    def bounds(self) -> list:
        return [(0.0, 2.0) for _ in range(self.N)] + [
            tuple(np.deg2rad((-50, 50))) for _ in range(self.N)
        ]

    def solve_for_setpoints(self, x) -> WindfarmSolution:
        setpoints = list((_x1, _x2) for _x1, _x2 in zip(x[: self.N], x[self.N :]))
        return self.windfarm(self.layout, setpoints)

    def objective_func(self, x):
        windfarm_sol = self.solve_for_setpoints(x)
        return -windfarm_sol.Cp()
