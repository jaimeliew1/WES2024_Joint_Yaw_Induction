"""
Hello Jaime. This shows how Ctprime varies with wind speed. What I actually need
is how does Ctprime vary with yaw misalignment for each wind speed (both above
and below rated conditions). 

So we will need to implement a K omega controller combined with a rotor speed
limiter and a power limiter so that both pitch and tip speed ratio are
calculated as a function of wind spped. Then we yaw the turbine.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import polars as pl
from foreach import foreach
from MITRotor.ReferenceTurbines import IEA15MW
from mitwindfarm import BEM, Layout, Windfarm, RotorSolution
from rich import print
from scipy.optimize import root_scalar

from WES2024 import utils
from WES2024.CustomRotors import BEMUnifiedMomentumLUT

FILESTEM = Path(__file__).stem

layout = Layout(np.array([0.0]), np.array([0.0]))


windfarm = Windfarm(rotor_model=BEM(IEA15MW(), momentum_model=BEMUnifiedMomentumLUT()))

# Calculated in separate optimisation (see WES2024.Generate._single_turbine_opt)
# Setpoints using unified momentum model
PITCH_OPT = -0.023146163628916267
TSR_OPT = 9.23061314763139
CP_OPT = 0.5072138998869634

# Radius of IEA15MW
R = 242.23775645 / 2

# max rotor speed in radians per second
rotor_speed_max = 0.7916813487046278

# Rated power of IEA15MW (W)
P_MAX = 15000000
air_density = 1.225


@dataclass
class Result:
    Uamb: float
    yaw: float
    pitch: float
    Cp: float
    Ct: float
    Ctprime: float


def evaluate_turbine(pitch_deg, tsr, yaw_deg) -> RotorSolution:
    return windfarm(layout, [(np.deg2rad(pitch_deg), tsr, np.deg2rad(yaw_deg))]).rotors[0]


def calc_turbine_setpoint(Uamb: float, yaw: float) -> Result:
    # Converge on tsr
    def func(x):
        tsr = x
        rot_sol = evaluate_turbine(np.rad2deg(PITCH_OPT), tsr, yaw)
        return rot_sol.Cp / rot_sol.extra.tsr**3 - CP_OPT / TSR_OPT**3

    root_sol = root_scalar(func, bracket=(3, TSR_OPT))

    # Limit tsr based on max rotor speed
    tsr = np.minimum(root_sol.root, R * rotor_speed_max / Uamb)
    rotor_sol = evaluate_turbine(np.rad2deg(PITCH_OPT), tsr, yaw)

    Cp = np.minimum(rotor_sol.Cp, 2 * P_MAX / (1.225 * np.pi * R**2 * Uamb**3))
    if Cp == CP_OPT:
        return Result(
            Uamb,
            yaw,
            np.rad2deg(PITCH_OPT),
            rotor_sol.Cp,
            rotor_sol.Ct,
            rotor_sol.Ctprime,
        )

    def func(x):
        pitch_deg = x
        sol = evaluate_turbine(pitch_deg, tsr, yaw)
        return sol.Cp - Cp

    root_sol = root_scalar(func, bracket=(np.rad2deg(PITCH_OPT), 30))
    rotor_sol = evaluate_turbine(root_sol.root, tsr, yaw)
    return Result(
        Uamb,
        yaw,
        root_sol.root,
        rotor_sol.Cp,
        rotor_sol.Ct,
        rotor_sol.Ctprime,
    )


def _generate(x) -> pl.DataFrame:
    U, yaw = x
    res = calc_turbine_setpoint(U, yaw)
    return pl.from_dict(asdict(res))


@utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False) -> pl.DataFrame:
    Us = np.arange(4, 25.1, 1)
    yaws = np.arange(-30, 30.1, 2)

    params = list(product(Us, yaws))
    results = foreach(_generate, params, context="spawn", parallel=True)

    df = pl.concat(results)

    return df


def plot(df: pl.DataFrame) -> None:

    fig, axes = plt.subplots(1, 3, sharex=True, figsize=2 * np.array([5, 2]))
    # plt.subplots_adjust(wspace=0)
    for key, ax in zip(["Cp", "Ct", "Ctprime"], axes):
        print(f"plotting {key}...")
        sns.lineplot(
            df.filter(pl.col("Uamb") < 30),
            x="yaw",
            y=key,
            hue="Uamb",
            ax=ax,
            palette="viridis",
            legend="full" if key == "Ctprime" else False,
        )
        ax.set_title(key)

    sns.move_legend(axes[-1], "center left", bbox_to_anchor=(1.1, 0.45))
    plt.savefig(utils.FIGDIR / "test_above_rate.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    df = generate(regenerate=False)
    print(df)
    plot(df)
