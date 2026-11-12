"""
LES precursor inflow of the 25-turbine farm as a mitwindfarm Windfield (ArbitraryZWindfield), to be passed to
R1.wakemodels.get_windfarm(..., inflow=...) instead of the uniform TIamb inflow or a power law.

The profiles are the x-y averages of the time-averaged precursor (Run04) of the LES, cropped below the sponge layer, from
LES_DATA/R1/abl_profiles.csv (generate/generate_abl_profiles.py):
    z      height / D, hub at z = 1   -> shifted so that the hub is at z = 0 (the turbine z in the wake models); the LES
                                         ground (z = 0) is then z = -1, i.e. where `bottom_wall_z=-1` puts the wall
    U      |<u>, <v>| / U_h            (hub-height speed U_h: 8.714 m/s, farm)
    phi    wind direction arctan2(<v>, <u>) in the simulation frame [rad] -- the angle to the turbine axis, which is the
           x axis of the farm. Not made hub-aligned by default: the hub-height direction is not exactly 0 in the LES
           (0.33 deg for the farm) and neither the turbines nor the model should pretend it is. `hub_aligned=True`
           subtracts the hub-height direction (veer only).
    TI     sqrt(2 k / 3) / U_h         resolved TKE k = (<u'u'> + <v'v'> + <w'w'>) / 2, divided by the HUB speed (so it is
                                         not a local turbulence intensity away from the hub). TI(z_hub) = 0.0462 for the
                                         farm, vs the TIAMB = 0.056 (sqrt(<u'u'>)/U_h) the models normally get.
np.interp holds the end values outside the profile (below the lowest cell centre z = 0.03 D, above the sponge).

    from WES2024.R1.get_les_inflow_windfarm import get_les_inflow_windfarm
    wf = get_windfarm("varvortex", rotor_model=rotor, inflow=get_les_inflow_windfarm())
"""

import numpy as np
import polars as pl
from mitwindfarm import ArbitraryZWindfield

from WES2024.R1.les_data import ABL_PROFILES

Z_HUB = 1.0  # hub height in diameters in the LES (generate_abl_profiles.Z_HUB)


def les_inflow(source: str = "farm", hub_aligned: bool = False) -> ArbitraryZWindfield:
    """ArbitraryZWindfield of the precursor profiles of `source` ("farm" or "two_turbine"), see module docstring."""
    d = pl.read_csv(ABL_PROFILES).filter(pl.col("source") == source).sort("z")
    z = d["z"].to_numpy()
    at = lambda col: float(np.interp(Z_HUB, z, d[col].to_numpy()))
    U_h = at("U")
    phi = np.deg2rad(d["phi"].to_numpy())
    if hub_aligned:
        phi = phi - np.deg2rad(at("phi"))
    return ArbitraryZWindfield(
        z=z - Z_HUB,
        U_z=d["U"].to_numpy() / U_h,
        wdir_z=phi,
        TIamb_z=np.sqrt(2.0 * d["tke"].to_numpy() / 3.0) / U_h,
    )


def get_les_inflow_windfarm(hub_aligned: bool = False) -> ArbitraryZWindfield:
    """Inflow of the 25-turbine (control_WES_R1 / square_NoControl) LES."""
    return les_inflow("farm", hub_aligned)


if __name__ == "__main__":
    w = get_les_inflow_windfarm()
    z = np.array([-0.5, 0.0, 0.5])
    print("z/D - z_hub     :", z)
    print("U / U_h         :", np.round(w.wsp(0, 0, z), 4))
    print("wind dir [deg]  :", np.round(np.rad2deg(w.wdir(0, 0, z)), 3))
    print("TI              :", np.round(w.TI(0, 0, z), 4))
