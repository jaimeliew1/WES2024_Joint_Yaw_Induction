from pathlib import Path

import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numpy.typing import ArrayLike
import numpy as np

from mitwindfarm.Layout import Layout
from mitwindfarm.windfarm import Windfarm, WindfarmSolution
import polars as pl

from cache import cache_polars

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)

windfarm = Windfarm()
layout = Layout([0, 6], [0.0, 0.0])


class NoControl:
    def __init__(self, layout, windfarm):
        self.layout = layout
        self.windfarm = windfarm

    def optimise(self) -> WindfarmSolution:
        setpoints = [(2, 0), (2, 0)]
        return self.windfarm(self.layout, setpoints)


class Controller:
    def __init__(self, layout, windfarm):
        self.layout = layout
        self.windfarm = windfarm

    def optimise(self) -> WindfarmSolution:
        x0 = self.initial_guess()
        bounds = self.bounds()
        sol = minimize(self.objective_func, x0, bounds=bounds)
        asdf = self.solve_for_setpoints(sol.x)
        # print(sol.x, asdf.setpoints[0][0])
        return asdf


class YawControl(Controller):
    def initial_guess(self) -> ArrayLike:
        return [0, 0]

    def bounds(self) -> list:
        return np.deg2rad([(-50, 50), (-50, 50)])

    def solve_for_setpoints(self, x) -> WindfarmSolution:
        setpoints = [(2, x[0]), (2, x[1])]
        return self.windfarm(self.layout, setpoints)

    def objective_func(self, x):
        windfarm_sol = self.solve_for_setpoints(x)
        return -windfarm_sol.Cp()


class ThrustControl(Controller):
    def initial_guess(self) -> ArrayLike:
        return [2.0, 2.0]

    def bounds(self) -> list:
        return [(0, 3), (0, 3)]

    def solve_for_setpoints(self, x) -> WindfarmSolution:
        setpoints = [(x[0], 0), (x[1], 0)]
        return self.windfarm(self.layout, setpoints)

    def objective_func(self, x):
        windfarm_sol = self.solve_for_setpoints(x)
        return -windfarm_sol.Cp()


class JointControl(Controller):
    def initial_guess(self) -> ArrayLike:
        return [2, 2, 0, 0]

    def bounds(self) -> list:
        return [(0, 2), (0, 2), np.deg2rad((-50, 50)), np.deg2rad((-50, 50))]

    def solve_for_setpoints(self, x) -> WindfarmSolution:
        setpoints = [(x[0], x[2]), (x[1], x[3])]
        return self.windfarm(self.layout, setpoints)

    def objective_func(self, x):
        windfarm_sol = self.solve_for_setpoints(x)
        return -windfarm_sol.Cp()


methods = {
    "NoControl": NoControl,
    "YawControl": YawControl,
    "ThrustControl": ThrustControl,
    "JointControl": JointControl,
}


@cache_polars(Path(__file__).parent.parent / "data/plot_01_2_turbine_wind_sweep.csv")
def generate(regenerate=False):
    data = []
    wdirs = np.arange(-20, 20, 0.25)
    for label, method in methods.items():
        for wdir in wdirs:
            sol = method(layout.rotate(wdir), windfarm).optimise()
            data.append(
                dict(
                    method=label,
                    wdir=wdir,
                    Cp=sol.Cp(),
                    Ctprime=float(sol.rotors[0].Ctprime),
                    yaw=np.rad2deg(sol.rotors[0].yaw),
                )
            )

    df = pl.from_dicts(data)
    return df


def plot(df):
    fig, axes = plt.subplots(3, 1, sharex=True)

    for method in methods:
        _df = df.filter(pl.col("method") == method).sort("wdir")
        axes[0].plot(_df["wdir"], _df["Cp"], label=method)
        axes[1].plot(_df["wdir"], _df["Ctprime"], label=method)
        axes[2].plot(_df["wdir"], _df["yaw"], label=method)

    axes[0].set_ylim(0, 0.60)
    axes[0].legend()

    plt.savefig(FIGDIR / "wind_direction_sweep.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=False)
    plot(df)


if __name__ == "__main__":
    main()
