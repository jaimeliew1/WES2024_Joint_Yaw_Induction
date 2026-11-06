"""
R1: the reduced LES data the paper figures read. These files are committed to the repository (LES_DATA/), so every figure can be
made without the raw LES (padeopsIO, /scratch). They are extracted from the raw LES by

    LES_DATA/twoturbine_alldata_LES_std.csv      two-turbine sweep: Cp, 10-min window statistics, ud (tmp/extra/twoturbine_add_les_std.py, not committed)
    LES_DATA/R1/twoturbine_rews.csv              two-turbine sweep: rotor-equivalent wind speed of the leading turbine
    LES_DATA/R1/uniform_inflow_les_delta_032.csv uniform-inflow LES the Unified Momentum Model was validated against (ctp, yaw [deg], ct, cp)
    LES_DATA/R1/abl_profiles.csv, abl_summary.csv  precursor profiles and inflow statistics (generate/generate_abl_profiles.py)
    LES_DATA/R1/les_farm_power_<key>.npz         10-min block power of the 5 x 5 farm runs (generate/generate_les_farm_power.py)
    LES_DATA/R1/les_hubplane_Joint_TANDEM.npz    hub-plane velocity of the Joint run (generate/generate_les_tandem_hubplane.py --les)
"""

from pathlib import Path

import polars as pl

LES_DATA = Path(__file__).parents[2] / "LES_DATA"
LES_DATA_R1 = LES_DATA / "R1"

TWOTURBINE_LES = LES_DATA / "twoturbine_alldata_LES_std.csv"
TWOTURBINE_REWS = LES_DATA_R1 / "twoturbine_rews.csv"
UNIFORM_INFLOW_LES = LES_DATA_R1 / "uniform_inflow_les_delta_032.csv"
ABL_PROFILES = LES_DATA_R1 / "abl_profiles.csv"
ABL_SUMMARY = LES_DATA_R1 / "abl_summary.csv"
HUBPLANE_JOINT = LES_DATA_R1 / "les_hubplane_Joint_TANDEM.npz"


def farm_power_path(key: str) -> Path:
    """Block power of the farm-LES run `key` of farm_control_cases.CASES."""
    return LES_DATA_R1 / f"les_farm_power_{key}.npz"


def load_twoturbine_lead(ctp: float) -> pl.DataFrame:
    """Leading turbine of the two-turbine ABL LES at one Ct', sorted by yaw [deg]. Cp_1 and Ct = Ct' (ud / REWS)^2 are normalised with REWS."""
    les = pl.read_csv(TWOTURBINE_LES).filter((pl.col("ctp") - ctp).abs() < 1e-6)
    df = les.select("casename", "yaw", "ud", "Cp_1").join(pl.read_csv(TWOTURBINE_REWS), on="casename")
    return df.with_columns(Ct=ctp * (pl.col("ud") / pl.col("rews")) ** 2).sort("yaw")
