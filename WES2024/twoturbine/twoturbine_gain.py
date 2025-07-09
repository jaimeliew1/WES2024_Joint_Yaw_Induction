"""
Compute two-turbine gain etc.

Kirby Heck
2024 Dec 10
"""

import polars as pl
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator as RGI

ROOTDIR = Path(__file__).parent
DATA_LES = ROOTDIR / "twoturbine_alldata_LES.csv"
DATA_UNIFIED = ROOTDIR / "twoturbine_mitwindfarm_unified.csv"
DATA_COSINE = ROOTDIR / "twoturbine_mitwindfarm_cosine.csv"
DATA_COSINE3 = ROOTDIR / "twoturbine_mitwindfarm_cosine3.csv"
DATA_JFM = ROOTDIR / "twoturbine_mitwindfarm_jfm.csv"


def interpolate_values(x, y, xp, yp, zp):
    """Interpolate `zp` with values xp, yp on a regular grid"""
    # Create the interpolator
    nx = len(xp.unique())
    ny = len(yp.unique())
    interpolator = RGI((xp.unique(), yp.unique()), np.array(zp).reshape((ny, nx)).T)

    # Perform the interpolation
    ret = interpolator(np.array((x, y)).T)
    return ret


def run():
    unified = pl.read_csv(DATA_UNIFIED)
    cosine = pl.read_csv(DATA_COSINE)
    cosine3 = pl.read_csv(DATA_COSINE3)
    jfm = pl.read_csv(DATA_JFM)
    les = pl.read_csv(DATA_LES)
    cp_max_les = les["Cp_T"].max()
    cp_betz_les = les.filter(ctp=2, yaw=0)["Cp_T"][0]

    model_dict = dict(LES=les, Unified=unified, Cosine=cosine, Cosine3=cosine3, JFM=jfm)

    for name, df in model_dict.items():
        cp_max = df["Cp_T"].max()
        ctp_betz = df["ctp"][int(np.argmin(abs(df["ctp"] - 2.0)))]
        cp_betz = df.filter(ctp=ctp_betz, yaw=0)["Cp_T"][0]
        cp_max_yaw = df.filter(ctp=ctp_betz)["Cp_T"].max()
        cp_max_thrust = df.filter(yaw=0)["Cp_T"].max()
        gain = (cp_max - cp_betz) / cp_betz
        gain_yaw = (cp_max_yaw - cp_betz) / cp_betz
        gain_thrust = (cp_max_thrust - cp_betz) / cp_betz

        # optimal setpoint:
        max_id = int(np.argmax(df["Cp_T"]))
        yaw_opt = df["yaw"][max_id]
        ctp_opt = df["ctp"][max_id]

        print(f"{name} baseline Cp_T: {cp_betz:.3f}")
        print(f"{name} gain: {gain * 100:.3f}%")
        print(f"{name} gain, yaw control: {gain_yaw * 100:.3f}%")
        print(f"{name} gain, thrust control: {gain_thrust * 100:.3f}%")

        # interpolate model set point onto LES optimal:
        LES_max = interpolate_values(yaw_opt, ctp_opt, les["yaw"], les["ctp"], les["Cp_T"])
        print(f"{name} (yaw, Ctp) = ({yaw_opt:.1f}, {ctp_opt:.1f})")
        print(
            f"{name} opt setpoints, LES gain Cp_T: {((LES_max - cp_betz_les) / cp_betz_les)[0] * 100:.3f}%"
        )


if __name__ == "__main__":
    run()
