"""
Inflow (precursor) profiles of the two-turbine and the 25-turbine LES, plus a table of inflow summary statistics.

The profiles are the x-y averages of the time-averaged precursor (Run04) budget terms, so they are exactly the
statistics of the inflow the turbines see. Each profile is cropped below the sponge layer (zstSponge * Lz, i.e. the
top 25% of the domain is dropped) and dimensionalized with the geostrophic wind G = Ro * Omega * D.

The summary table lists, per simulation:
    U_h     hub-height wind speed |<u>, <v>| in m/s
    TI_h    sqrt(2k/3) / U_h at hub height, k = resolved TKE = (<u'u'> + <v'v'> + <w'w'>) / 2 (no SGS contribution)
    alpha   power-law shear exponent, U ~ z^alpha, least-squares fit in log-log space to all levels from the surface
            up to the rotor top tip (z <= z_hub + D/2)
    veer    wind direction at the rotor bottom tip minus at the rotor top tip, in degrees (positive = clockwise turning
            with height, i.e. veering in the northern hemisphere)
    u*      friction velocity from the run log (BudgetIO.get_ustar), dimensionalized with G
    TI_u    streamwise turbulence intensity sqrt(<u'u'>) / U_h, for reference: this is the definition behind the
            ambient TI = 0.056 that the wake models are given (WES2024.LES.shared.TIAMB)

Output: LES_DATA/R1/abl_profiles.csv (profiles, long format) and LES_DATA/R1/abl_summary.csv (table).

Usage:  python -m WES2024.R1.generate.generate_abl_profiles
"""

import numpy as np
import padeopsIO as pio
import polars as pl

from WES2024.R1.les_data import ABL_PROFILES as FILE_PROFILES, ABL_SUMMARY as FILE_SUMMARY

D = 100.0  # m, rotor diameter in all simulations
OMEGA = 7.29e-5  # rad/s, rotation rate of the Earth
Z_HUB = 1.0  # hub height in diameters

# LES whose precursor (Run04) supplies the inflow. All two-turbine cases share the same statistical inflow, so a single
# one is used. The 25-turbine precursor is the same for all the control strategies; the no-control case is used.
SOURCES = {
    "two_turbine": dict(
        label="Two-turbine LES",
        path="/scratch/08445/tg877441/control_twoturbine/export/yaw_00_ct_01",
    ),
    "farm": dict(
        label="25-turbine LES",
        path="/scratch/08445/tg877441/control_WES_R1/square_NoControl",
    ),
}
PRECURSOR_RUNID = 4


def read_profiles(path):
    """Dimensional, sponge-cropped inflow profiles of one LES. Returns (profile dataframe, G, ustar, sim)."""
    sim = pio.BudgetIO(path, padeops=True, runid=PRECURSOR_RUNID, verbose=False)
    G = sim.Ro * OMEGA * D  # Ro = G / (Omega D)

    ds = sim.xy_avg(budget_terms=["ubar", "vbar", "uu", "vv", "ww", "Tbar"])
    z = ds["z"].to_numpy()

    # top of the domain: the sponge acts above zstSponge * Lz
    Lz = z[-1] + 0.5 * (z[1] - z[0])
    zst = float(pio.utils.io_utils.key_search_r(sim.input_nml, "zstsponge"))
    keep = z <= zst * Lz

    u, v = ds["ubar"].to_numpy(), ds["vbar"].to_numpy()
    tke = 0.5 * (ds["uu"].to_numpy() + ds["vv"].to_numpy() + ds["ww"].to_numpy())  # uu etc. are covariances

    df = pl.DataFrame(
        {
            "z": z,  # diameters
            "U": np.hypot(u, v) * G,  # m/s
            "phi": np.rad2deg(np.arctan2(v, u)),  # deg, in the simulation frame
            "tke": tke * G**2,  # m^2/s^2
            "uu": ds["uu"].to_numpy() * G**2,  # m^2/s^2
            "theta": ds["Tbar"].to_numpy(),  # K
        }
    ).filter(pl.Series(keep))
    return df, G, float(sim.get_ustar()) * G, Lz, zst


def summarize(df, G, ustar, label):
    """Hub-height speed and TI, shear exponent up to the rotor top tip, veer across the rotor, friction velocity."""
    z = df["z"].to_numpy()
    at = lambda col, zq: float(np.interp(zq, z, df[col].to_numpy()))

    U_h = at("U", Z_HUB)
    TI_h = np.sqrt(2 * at("tke", Z_HUB) / 3) / U_h
    TI_u = np.sqrt(at("uu", Z_HUB)) / U_h

    below_tip = z <= Z_HUB + 0.5
    alpha = np.polyfit(np.log(z[below_tip] / Z_HUB), np.log(df["U"].to_numpy()[below_tip]), 1)[0]
    in_rotor = below_tip & (z >= Z_HUB - 0.5)
    alpha_rotor = np.polyfit(np.log(z[in_rotor] / Z_HUB), np.log(df["U"].to_numpy()[in_rotor]), 1)[0]

    veer = at("phi", Z_HUB - 0.5) - at("phi", Z_HUB + 0.5)
    return dict(
        simulation=label, G=G, U_h=U_h, TI_h=TI_h, TI_u=TI_u, alpha=alpha, alpha_rotor=alpha_rotor, veer=veer, ustar=ustar,
    )


def run():
    profiles, rows = [], []
    for key, src in SOURCES.items():
        print(f"reading {key}: {src['path']}")
        df, G, ustar, Lz, zst = read_profiles(src["path"])
        print(f"  G = {G:.3f} m/s, Lz = {Lz:.2f} D, sponge above {zst:.2f} Lz = {zst * Lz:.2f} D, {df.height} levels kept")
        profiles.append(df.with_columns(source=pl.lit(key)))
        rows.append(summarize(df, G, ustar, src["label"]) | dict(source=key, Lz=Lz, z_sponge=zst * Lz))

    pl.concat(profiles).write_csv(FILE_PROFILES)
    summary = pl.DataFrame(rows)
    summary.write_csv(FILE_SUMMARY)
    return summary


def print_table(summary):
    print()
    print(f"{'Simulation':<18}{'U_h [m/s]':>10}{'TI_h [-]':>10}{'alpha [-]':>10}{'veer [deg]':>12}{'u* [m/s]':>10}{'G [m/s]':>9}{'TI_u [-]':>10}")
    for r in summary.iter_rows(named=True):
        print(
            f"{r['simulation']:<18}{r['U_h']:>10.3f}{r['TI_h']:>10.4f}{r['alpha']:>10.3f}{r['veer']:>12.2f}"
            f"{r['ustar']:>10.4f}{r['G']:>9.2f}{r['TI_u']:>10.4f}"
        )
    print("alpha: power-law fit from the surface up to the rotor top tip (z <= 1.5 D).")
    print("alpha fitted over the rotor layer only (0.5 D <= z <= 1.5 D): " + ", ".join(
        f"{r['simulation']} {r['alpha_rotor']:.3f}" for r in summary.iter_rows(named=True)
    ))
    print("veer: wind direction at 0.5 D minus at 1.5 D (positive = clockwise turning with height).")
    print("TI_h uses the resolved TKE only. TI_u = sqrt(<u'u'>)/U_h (the definition of the models' TIAMB = 0.056).")


if __name__ == "__main__":
    print_table(run())
