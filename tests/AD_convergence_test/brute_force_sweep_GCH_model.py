from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
import seaborn as sns
from foreach import foreach
from WES2024 import utils
from WES2024.optimise import JointControl, NoControl, ThrustControl, YawControl, CTPRIME_OPT
from WES2024.LES.shared import TIAMB

from mitwindfarm import Layout, Area
from gch.Windfarm import Windfarm
from gch.Windfield import Uniform

from gch.Superposition import FreestreamQuadratic
from gch.Rotor import UnifiedLUTAD
# from gch.optimize import JointControl
from gch.RotorGrid import FlorisGrid
from gch.Wake import GaussCurlHybridModel

import matplotlib.pyplot as plt

FILESTEM = Path(__file__).stem


yaws = np.radians([0, 10, 20, 30, 40, 50])
colors = sns.color_palette("magma", n_colors=len(yaws))
wdir = -3
layout = Layout([0, 4], [0.0, 0.0], [80/90.0, 80/90.0]).rotate(wdir)


windfarm = Windfarm(
    rotor_model=UnifiedLUTAD(
    # rotor_grid=FlorisGrid(),
    rotor_grid=Area()
    ),
    wake_model=GaussCurlHybridModel(
    # rotor_grid=FlorisGrid()
    rotor_grid=Area()
    ),
    base_windfield=Uniform(TIamb=TIAMB),
    TIamb=TIAMB,
)

def _solve_for_params(x):
    ctprime, yaw = x
    sol = windfarm(layout, [(ctprime, yaw), (CTPRIME_OPT, 0.0)])

    return pl.DataFrame(
        {
            "ctprime": [ctprime],
            "yaw": [yaw],
            "wdir": [wdir],
            "Cp": [sol.Cp],
            "Cp_1" : [sol.rotors[0].Cp],
            "Cp_2" : [sol.rotors[1].Cp],
            'Ct_1' : [sol.rotors[0].Ct],
        }
    )

@utils.cache_polars(utils.CACHEDIR / f'{FILESTEM}.csv')
def brute_force_sweep():
    ctprimes = np.linspace(0.5, 10.0, 100)
    param_grid = list(product(ctprimes, yaws))
    results = foreach(_solve_for_params, param_grid)
    results_df = pl.concat(results)
    return results_df

def optimize():
    sol = JointControl(layout, windfarm).optimise(use_gradients=False)
    return pl.DataFrame(
        {
            "ctprime": [sol.rotors[0].Ctprime],
            "yaw": [np.degrees(sol.rotors[0].yaw)],
            "wdir": [wdir],
            "Cp": [sol.Cp],
            "Cp_1" : [sol.rotors[0].Cp],
            "Cp_2" : [sol.rotors[1].Cp],
        }
    )


def plot():
    df_bf = brute_force_sweep(regenerate=False)
    fig, ax = plt.subplots(figsize = (6, 4.5))
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
    ax.set_title("Brute force sweep, GCH wake model")
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

    fig, ax = plt.subplots(figsize = (6, 4.5))
    for i, yaw in enumerate(yaws):
         df_slice = df_bf.filter(pl.col("yaw") == yaw).sort("ctprime")
         ax.plot(
             df_slice["ctprime"],
             df_slice["Ct_1"],
             label=f"$\\gamma$={np.degrees(yaw):.0f}°",
             color = colors[i]
         )
    # black dashed line at Ct_1 = 1
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel("$C_{T}'$")
    ax.set_ylabel("$C_{T,1}$")
    ax.set_title("Upstream turbine thrust coefficient, GCH wake model")
    ax.legend(frameon=False)
    plt.savefig(utils.FIGDIR / f"{FILESTEM}_upstream_turbine_ct.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    df = brute_force_sweep(regenerate=True)
    print(df)
    plot()
    opt_df = optimize()
    print(opt_df)


