from pathlib import Path

import numpy as np
import polars as pl
from foreach import foreach
from MITRotor.BEMSolver import BEM, BEMSolution
from MITRotor.ReferenceTurbines import IEA15MW
from scipy.interpolate import BSpline, make_interp_spline
from scipy.optimize import minimize, minimize_scalar, root_scalar

from WES2024 import utils

__all__ = ["generate"]

REGENERATE = True

FILESTEM = Path(__file__).stem

YAWS = np.arange(0.0, 50.1, 2.5)

rotor = IEA15MW()
bem = BEM(rotor=rotor)


def find_optimal_setpoint(bem: BEM, yaw: float = 0) -> BEMSolution:
    """
    Finds the optimal pitch and tip speed ratio (tsr) setpoint for a wind
    turbine at a given yaw misalignment angle.

    Parameters:
    - bem (BEM): The Blade Element Momentum model.
    - yaw (float): Yaw misalignment angle in radians (default is 0).

    Returns:
    BEMSolution: The optimal BEM solution.
    """

    def to_opt(x):
        pitch, tsr = x
        return -bem(pitch, tsr, yaw).Cp()

    res = minimize(to_opt, (0, 9))
    pitch, tsr = res.x
    print(yaw, res)
    return bem(pitch, tsr, yaw)


def Cp_isobars(Cp_target: float, bem: BEM, sol_opt: BEMSolution, N_theta: int = 20) -> list:
    """
    Generates points on constant Cp isobars for a given target Cp using the
    Blade Element Momentum (BEM) model.

    Parameters:
    - Cp_target (float): Target coefficient of power.
    - bem (BEM): The Blade Element Momentum model.
    - sol_opt (BEMSolution): The optimal solution for the BEM model.
    - N_theta (int): Number of angles to consider (default is 20).

    Returns:
    list: A list of tuples containing pitch, tsr, Cp, and Ctprime on the Cp isobars.
    """

    x0 = np.array([sol_opt.pitch, sol_opt.tsr])
    points = []
    angles = np.linspace(0.05, 0.7 * np.pi, N_theta)
    for angle in angles:
        dx = np.array([0.1 * np.cos(angle), -3 * np.sin(angle)])

        def to_opt(c):
            setpoint = x0 + c * dx

            return Cp_target - bem(*setpoint, sol_opt.yaw).Cp()

        res = root_scalar(to_opt, x0=0, x1=1, maxiter=20)

        if res.converged:
            sol = bem(*(x0 + res.root * dx), sol_opt.yaw)
            points.append((sol.pitch, sol.tsr, sol.Cp(), sol.Ctprime()))

    return points


def Ctprime_minimising_Cp_setpoint(
    Cp_target: float, bem: BEM, sol_opt: BEMSolution, N_theta: int = 20
) -> BEMSolution:
    """
    Finds the pitch and tip speed ratio (tsr) setpoint that minimizes Ctprime on
    a constant Cp isobar.

    Parameters:
    - Cp_target (float): Target coefficient of power.
    - bem (BEM): The Blade Element Momentum model.
    - sol_opt (BEMSolution): The optimal solution for the BEM model.
    - N_theta (int): Number of angles to consider (default is 20).

    Returns:
    BEMSolution: The solution with minimized Ctprime on the specified Cp isobar.
    """
    Cp_isobar_points = Cp_isobars(Cp_target, bem, sol_opt, N_theta=N_theta)
    spline = make_interp_spline(np.linspace(0, 1, len(Cp_isobar_points)), Cp_isobar_points)
    res = minimize_scalar(lambda x: spline(x)[3], bounds=(0, 1))
    pitch_opt, tsr_opt, _, _ = spline(res.x)
    return bem(pitch_opt, tsr_opt, sol_opt.yaw)


def generate_derate_strat(bem: BEM, yaw: float, N_Cp: int = 10, N_theta: int = 20) -> pl.DataFrame:
    """
    Generates the Ct-minimising derate strategy trajectory for a given BEM model
    at a given yaw angle.

    Parameters:
    - bem (BEM): The Blade Element Momentum model.
    - yaw (float): Yaw misalignment angle in radians.
    - N_Cp (int): Number of Cp values to consider (default is 10).
    - N_theta (int): Number of angles to consider (default is 20).

    Returns:
    pl.DataFrame: A Polars DataFrame containing derate strategy data, including
    Cp, derate, pitch, tsr, and Ctprime.
    """

    sol_opt: BEMSolution = find_optimal_setpoint(bem, yaw)
    Cp_opt = sol_opt.Cp()

    Cps = np.linspace(0.35 * Cp_opt, Cp_opt - 0.01, N_Cp)

    trajectory = []
    for Cp in Cps:
        sol = Ctprime_minimising_Cp_setpoint(Cp, bem, sol_opt, N_theta=N_theta)
        trajectory.append((sol.pitch, sol.tsr, Cp, sol.Ctprime()))

    trajectory.append((sol_opt.pitch, sol_opt.tsr, Cp_opt, sol_opt.Ctprime()))
    Cps = np.concatenate([Cps, [Cp_opt]])

    derate_factors = [Cp / Cp_opt for Cp in Cps]

    out = []
    for Cp, derate, p in zip(Cps, derate_factors, trajectory):
        out.append(dict(Cp=Cp, derate=derate, pitch=p[0], tsr=p[1], Ctprime=p[3]))
    return pl.from_dicts(out)


def derate_spline(bem: BEM, yaw: float, N_Cp: int = 10, N_theta: int = 20) -> BSpline:
    """
    Generates a spline representation of the Ct-minimising derate strategy for a
    given BEM model and yaw misalignment.

    Parameters:
    - bem (BEM): The Blade Element Momentum model.
    - yaw (float): Yaw misalignment angle in radians.
    - N_Cp (int): Number of Cp values to consider (default is 10).
    - N_theta (int): Number of angles to consider (default is 20).

    Returns:
    BSpline: A B-spline representation of the derate strategy.
    """
    df = generate_derate_strat(bem, yaw, N_Cp=N_Cp, N_theta=N_theta)
    derate = df["derate"].to_numpy()
    points = df.select(["pitch", "tsr", "Cp", "Ctprime"])
    spline = make_interp_spline(derate, points)

    return spline


def _generate(x):
    yaw = x
    return generate_derate_strat(bem, np.deg2rad(yaw), N_Cp=20).with_columns(yaw=yaw)


@utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False) -> pl.DataFrame:

    out = foreach(_generate, YAWS, parallel=True)
    out = pl.concat(out)

    return out


if __name__ == "__main__":
    df = generate(regenerate=REGENERATE)
    print(df)
