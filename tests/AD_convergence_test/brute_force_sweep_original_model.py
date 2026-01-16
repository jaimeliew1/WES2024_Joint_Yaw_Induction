from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
import seaborn as sns
from foreach import foreach
from WES2024 import utils
from WES2024.optimise import JointControl, NoControl, ThrustControl, YawControl, CTPRIME_OPT
from WES2024.LES.shared import TIAMB

from mitwindfarm import Layout, Area, Windfarm, Niayifar
from WES2024.CustomRotors import UnifiedLUTAD
from WES2024.LES_new.final_calibration import get_wakemodel

import matplotlib.pyplot as plt

FILESTEM = Path(__file__).stem


yaws = np.radians([0, 10, 20, 30, 40, 50])
colors = sns.color_palette("magma", n_colors=len(yaws))


windfarm = Windfarm(
    rotor_model=UnifiedLUTAD(), 
    wake_model=get_wakemodel(), 
    TIamb=TIAMB,
    superposition=Niayifar(),
)

def _solve_for_params(x):
    ctprime, yaw, wdir = x
    layout = Layout([0, 4], [0.0, 0.0], [80/90.0, 80/90.0]).rotate(wdir)
    sol = windfarm(layout, [(ctprime, yaw), (CTPRIME_OPT, 0.0)])

    return pl.DataFrame(
        {
            "ctprime": [ctprime],
            "yaw": [yaw],
            "wdir": [wdir],
            "Cp": [sol.Cp],
            "Cp_1": [sol.rotors[0].Cp],
            "Cp_2": [sol.rotors[1].Cp],
        }
    )

@utils.cache_polars(utils.CACHEDIR / f'{FILESTEM}.csv')
def brute_force_sweep():
    ctprimes = np.linspace(0.5, 10.0, 100)

    # wdirs = np.arange(-20, 20, 1.0)
    wdirs = np.radians([3])

    param_grid = list(product(ctprimes, yaws, wdirs))
    results = foreach(_solve_for_params, param_grid)
    results_df = pl.concat(results)
    return results_df


def plot():
    df_bf = brute_force_sweep(regenerate=False)
    fig, ax = plt.subplots(figsize = (6, 4.5))
    # for wdir in [-10, 0, 10]:
    for i, yaw in enumerate(yaws):
         df_slice = df_bf.filter(pl.col("yaw") == yaw).sort("ctprime")
         ax.plot(
             df_slice["ctprime"],
             df_slice["Cp"],
             # c=df_slice["Cp"],
             label=f"$\\gamma$={np.degrees(yaw):.0f}°",
             # cmap="viridis",
             color = colors[i]
         )

    ax.set_xlabel("$C_{T}'$")
    ax.set_ylabel("$C_{P}$")
    ax.set_title("Brute force sweep, original wake model")
    ax.legend(frameon=False)
    plt.savefig(utils.FIGDIR / f"{FILESTEM}_bruteforce.png", dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots(figsize = (6, 4.5))
    for i, yaw in enumerate(yaws):
         df_slice = df_bf.filter(pl.col("yaw") == yaw).sort("ctprime")
         ax.plot(
             df_slice["ctprime"],
             df_slice["Cp_1"],
             label=f"$\\gamma$={np.degrees(yaw):.0f}°",
             color = colors[i]
         )
    ax.set_xlabel("$C_{T}'$")
    ax.set_ylabel("$C_{P,1}$")
    ax.set_title("Upstream turbine, GCH wake model")
    ax.legend(frameon=False)
    plt.savefig(utils.FIGDIR / f"{FILESTEM}_upstream_turbine.png", dpi=300, bbox_inches="tight") 


    fig, ax = plt.subplots(figsize = (6, 4.5))
    for i, yaw in enumerate(yaws):
         df_slice = df_bf.filter(pl.col("yaw") == yaw).sort("ctprime")
         ax.plot(
             df_slice["ctprime"],
             df_slice["Cp_2"],
             label=f"$\\gamma$={np.degrees(yaw):.0f}°",
             color = colors[i]
         )

    ax.set_xlabel("$C_{T}'$")
    ax.set_ylabel("$C_{P,2}$")
    ax.set_title("Downstream turbine, GCH wake model")
    ax.legend(frameon=False)
    plt.savefig(utils.FIGDIR / f"{FILESTEM}_downstream_turbine.png", dpi=300, bbox_inches="tight") 


if __name__ == "__main__":
    df = brute_force_sweep(regenerate=True)
    print(df)
    plot()


