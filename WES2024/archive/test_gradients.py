from itertools import product
from pathlib import Path
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from cache import cache_polars
from dualitic import DualVariables
from foreach import foreach
from mitwindfarm.Layout import Layout
from mitwindfarm.windfarm import Windfarm, WindfarmSolution
from numpy.typing import ArrayLike
from optimise import JointControl as JointControl_FD
from scipy.optimize import minimize
from utilities import to_polars
from rich import print

from plot_02_spiral import Spiral


class Controller:
    def __init__(self, layout, windfarm):
        self.layout = layout
        self.windfarm = windfarm
        self.N = len(layout)

    def optimise(self, verbose=False) -> WindfarmSolution:
        x0 = self.initial_guess()
        # breakpoint()
        bounds = self.bounds()
        sol = minimize(self.objective_func, x0, bounds=bounds, jac=True)
        if verbose:
            print(f"{sol.nfev=}")
            print(f"{sol.njev=}")
        asdf = self.solve_for_setpoints(sol.x)
        return asdf


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
        x = DualVariables(x)
        windfarm_sol = self.solve_for_setpoints(x)
        Cp = windfarm_sol.Cp().real[0]
        Cp_grad = windfarm_sol.Cp().dual[0]
        return -Cp, -Cp_grad


if __name__ == "__main__":
    # layout = Layout([0, 4, 8], [0, 0.5, 1.0])

    # x = DualVariables([2, 2, 2, 0, 0, 0])
    # setpoints = np.array(x).reshape([2, -1]).T
    windfarm = Windfarm()

    # sol = windfarm(layout, setpoints)
    # print(sol)

    # sol_opt = JointControl(layout, windfarm).optimise(verbose=True)
    # print(sol_opt)

    layout = Spiral(20, min_dist=4)

    tstart = perf_counter()
    sol_opt_grad = JointControl(layout, windfarm).optimise(verbose=True)
    print(sol_opt_grad.Cp(), t_grad := perf_counter() - tstart)

    tstart = perf_counter()
    sol_opt_FD = JointControl_FD(layout, windfarm).optimise(verbose=True)
    print(sol_opt_FD.Cp(), t_FD := perf_counter() - tstart)

    print((t_FD - t_grad) / t_FD)
