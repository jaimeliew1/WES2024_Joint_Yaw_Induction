"""
Compare turbine-level errors in calibration

Kirby Heck
2024 December 19
"""

from pathlib import Path
import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from UnifiedMomentumModel import Momentum

from WES2024.LES_new.compare_superposition import (
    CALIBRATION_RESULTS,
    BASE_LAYOUT,
    CALIBRATION_LAYOUT,
    ROW_INDICES,
    ROW_INDICES_CALIBRATION,
    read_LES_outputs,
    LES_output_dir,
    LES_pnormfact,
    wakemodels,  # need to import wake models from the other file; names must match
)
from WES2024.LES.shared import TIAMB
from WES2024.CustomRotors import UnifiedLUTAD
# from mitwindfarm import (
#     VariableKwGaussianWakeModel,
#     Niayifar,
#     Windfarm,
# )
from WES2024.LES_new.UpfalCustom.CustomWake import VariableKwGaussianWakeModel
from WES2024.LES_new.UpfalCustom.CustomRotors import AD
from WES2024.LES_new.UpfalCustom.CustomWindfarm import Windfarm
from WES2024.LES_new.UpfalCustom.CustomSuperposition import Niayifar

model_pnormfact = (16/27)  # Betz
uni = Momentum.UnifiedMomentum()
model_pnormfact = uni(2, 0).Cp

figpath = Path(__file__).parent / "calibration"


def plot_scatter_with_labels(x, y, data, text="{:.3f}", ax=None, **plt_kwargs):
    """
    Plot scatter data on the color axis at x, y locations

    Returns
    -------
    im, ax
    """
    if ax is None:
        _, ax = plt.subplots()

    im = ax.scatter(x, y, c=data, **plt_kwargs)
    if text is not None:
        for _x, _y, _c in zip(x, y, data):
            ax.text(_x, _y + 0.1, text.format(_c), ha="center", va="bottom")

    ax.set_aspect("equal")
    ax.set_xlabel("$x/D$ (-)")
    ax.set_ylabel("$y/D$ (-)")

    return im, ax


def plot_and_save(model_Cp, LES_Cp, fout, title=None):
    """Plots the base layout turbine errors"""
    err_normalized = (model_Cp - LES_Cp) / LES_Cp
    im, ax = plot_scatter_with_labels(
        BASE_LAYOUT.x,
        BASE_LAYOUT.y,
        err_normalized * 100,
        # df.filter(method=LES_case)['turbine_id'] - np.arange(25),
        text="{:+.1f}\\%",
        clim=[-25, 25],
        cmap="seismic",
        lw=1,
        edgecolor="k",
    )
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Relative error in $C_P$")
    plt.savefig(figpath / fout, dpi=300)
    plt.close()
    err = (np.mean(model_Cp) - np.mean(LES_Cp)) / np.mean(LES_Cp)
    print("Saved ", fout.name)
    print(f"  Normalized farm power error: {err*100:.1f}%")


def run(regenerate=False, wakemodel="04_kw_TI", method="nocontrol"):
    """Run script"""
    # load saved data
    df = read_LES_outputs(LES_output_dir.glob("*.csv"))
    df_cached = pl.read_json(CALIBRATION_RESULTS / "apriori_err.json")

    calibration = wakemodels[wakemodel](BASE_LAYOUT, ROW_INDICES)

    for LES_case in ['jointcontrol', 'yawcontrol']: #df['method'].unique():
        if LES_case == "calibration_18x3":
            continue

        fout = figpath / f"Cp_turbine_err_{LES_case}.png"
        if regenerate or not fout.exists():
            # assign set points from LES
            control_setpoints = df.filter(method=LES_case).select("Ctprime", "yaw").to_numpy()
            # extract wake model parameters
            params = np.array(df_cached.filter(method=method, wakemodel=wakemodel)["params"].to_list()).squeeze()
            # run MITWindfarm
            sol = calibration.run_windfarm(params, control_setpoints)
            LES_Cp = df.filter(method=LES_case)["Cp"].to_numpy() / LES_pnormfact
            model_Cp = np.array([rotor.Cp for rotor in sol.rotors]) / model_pnormfact

            plot_and_save(model_Cp, LES_Cp, fout, title=LES_case)

    print("Done")


def check_custom_params(params, wakemodel="04_kw_TI", LES_case=None, regenerate=True):
    """Custom set points from Ilan"""
    df = read_LES_outputs(LES_output_dir.glob("*.csv"))

    # set up MITWindfarm
    calibration = wakemodels[wakemodel](BASE_LAYOUT, ROW_INDICES)

    LES_case = LES_case or "nocontrol"

    fout = figpath / f"Cp_turbine_err_{LES_case}_custom.png"
    if not fout.exists() or regenerate:
        # assign set points from LES
        control_setpoints = df.filter(method=LES_case).select("Ctprime", "yaw").to_numpy()
        # run MITWindfarm
        sol = calibration.run_windfarm(params, control_setpoints)
        LES_Cp = df.filter(method=LES_case)["Cp"].to_numpy() / LES_pnormfact
        model_Cp = np.array([rotor.Cp for rotor in sol.rotors]) / model_pnormfact

        plot_and_save(model_Cp, LES_Cp, fout, title=LES_case)

    print("Done")


def check_verycustom_params(params, LES_case="nocontrol", regenerate=True):
    """Check Ilan's wake model parameters with Ilans' wake model"""
    wakemodel = VariableKwGaussianWakeModel(**params)
    windfarm = Windfarm(
        rotor_model=AD(), wake_model=wakemodel, TIamb=TIAMB, superposition=Niayifar()
    )

    # load LES data
    df = read_LES_outputs(LES_output_dir.glob("*.csv"))

    fout = figpath / f"Cp_turbine_err_{LES_case}_verycustom.png"
    if not fout.exists() or regenerate:
        # assign set points from LES
        control_setpoints = df.filter(method=LES_case).select("Ctprime", "yaw").to_numpy()
        # run MITWindfarm
        sol = windfarm(BASE_LAYOUT, control_setpoints)
        model_Cp = np.array([rotor.Cp for rotor in sol.rotors]).squeeze() / model_pnormfact
        LES_Cp = df.filter(method=LES_case)["Cp"].to_numpy() / LES_pnormfact

        plot_and_save(model_Cp, LES_Cp, fout, title=LES_case)

    print("Done")


if __name__ == "__main__":
    run(regenerate=True, method="nocontrol", wakemodel='06_kw_TI_Ctp')
    # check_custom_params([0.6111, 0, 1.996e-3], LES_case="nocontrol")
    # check_verycustom_params(dict(a=0.6111, b=0, c=1.996e-3), LES_case="nocontrol")
    print("Script complete")
