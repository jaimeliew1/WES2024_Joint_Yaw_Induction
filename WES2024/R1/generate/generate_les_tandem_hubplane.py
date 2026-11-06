# dualitic is imported first to ensure correct monkey patching.
import dualitic

"""
R1: hub-height velocity field of the 5x5 farm LES under joint control with the TANDEM set points, next to the TANDEM forward solve of the
same turbines. This is the TANDEM counterpart of the MITWindfarm-vs-LES schematic of the original paper.

Case: Joint_TANDEM of farm_control_cases.CASES (wdir 0, 6D x 6D 5x5 farm, Joint set points optimised with TANDEM + ground).

    LES     time-mean u (budget term "ubar", Run05, last cumulative dump) at the hub plane (z = 1 D, linearly interpolated between the two cell
            centres that bracket it), divided by the hub-height speed U_h of the run's own precursor (Run04), with the rotors from the ADM
            inputs turbines/ActuatorDisk_*_input.inp. Stored in LES_DATA/R1 (les_data.HUBPLANE_JOINT); `--les` re-reads it from the raw LES
            (padeopsIO, /scratch).
    TANDEM  wakemodels.get_windfarm("tandem") (with the ground) with the coupled vortex-decay model and the LES precursor profiles as inflow,
            so u / U_h = 1 at the hub far upstream. The rotors are run exactly as the LES ADM runs them: UnifiedLUTAD with the Ct' and yaw of the
            ADM inputs -- the same numbers as setpoints.csv (Ctprime, yaw), i.e. the BEM set points of the optimiser expressed as the (Ct', yaw)
            the LES can apply. The model field is the analytic base flow plus the marched u-deficit du, interpolated to the LES hub-plane grid.
            Cached in data/generate_les_tandem_hubplane.npz.

Coordinates are in diameters, shifted so that the first turbine (row 0, column 0) is at the origin, x streamwise, y spanwise. Yaws are positive
counter-clockwise seen from above, as the LES ADM `yaw`.

Usage: python -m WES2024.R1.generate.generate_les_tandem_hubplane [--les]
"""

import re
import sys
from pathlib import Path

import mitwindfarm as mitwf
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from WES2024 import utils
from WES2024.R1.farm_control_cases import les_path
from WES2024.R1.get_les_inflow_windfarm import get_les_inflow_windfarm
from WES2024.R1.les_data import HUBPLANE_JOINT
from WES2024.R1.wakemodels import get_windfarm

CASE = "Joint_TANDEM"
Z_HUB = 0.0  # all z-axes are normalized to hub height
BOX = dict(xlim=(-6.0, 40.0), ylim=(-6.0, 30.0))  # [D] relative to turbine 0, a few D around the farm
MODEL_KWARGS = dict(v_model="decay", v_kwargs=dict(couple=True))

CACHE = utils.CACHEDIR / f"{Path(__file__).stem}.npz"


def read_turbines(path):
    """(x, y, Ct', yaw [rad]) of every rotor from the ADM input files the LES was run with, ordered by turbine number."""
    rows = []
    for f in sorted(Path(path, "turbines").glob("ActuatorDisk_*_input.inp")):
        text = open(f).read()
        g = lambda k: float(re.search(k + r"\s*=\s*([-0-9.eEdD+]+)", text).group(1).lower().replace("d", "e"))
        rows.append((g("xLoc"), g("yLoc"), g("cT"), np.deg2rad(g("yaw"))))
    return np.array(rows)


def read_les():
    """Dict stored in LES_DATA/R1: x, y (LES grid [D]), u (nx, ny) [u / U_h], turbines (n, 4) = x, y, Ct', yaw [rad] relative to turbine 0, uhub, tidx, ..."""
    import padeopsIO as pio

    path = les_path(CASE)
    turbines = read_turbines(path)
    turbines[:, :2] -= turbines[0, :2]
    sim = pio.BudgetIO(str(path), padeops=True, runid=5, normalize_origin="turb")
    pre = pio.BudgetIO(str(path), padeops=True, runid=4, normalize_origin=sim.origin)
    uhub = pre.get_uhub()
    s = sim.slice(budget_terms=["ubar"], xlim=list(BOX["xlim"]), ylim=list(BOX["ylim"]), zlim=[Z_HUB - 0.1, Z_HUB + 0.1])
    z = s.z.values
    u = s["ubar"].values  # (x, y, z)
    k = np.searchsorted(z, Z_HUB)  # z[k-1] < Z_HUB <= z[k]
    w = (Z_HUB - z[k - 1]) / (z[k] - z[k - 1])
    return dict(x=s.x.values, y=s.y.values, u=((1 - w) * u[:, :, k - 1] + w * u[:, :, k]) / uhub, turbines=turbines, uhub=uhub,
                tidx=int(sim.budget_tidx), budget_n=int(sim.budget_n), t_budget_start=float(pio.key_search_r(sim.input_nml, "time_budget_start")))


def tandem_hubplane(T, x, y):
    """(u / U_h (nx, ny), Cp (n,)) of TANDEM at the hub plane (z = 0 in the model frame) on the (x, y) grid [D, relative to T[0]]."""
    layout = mitwf.Layout(T[:, 0], T[:, 1], np.zeros(len(T)))
    wf = get_windfarm("tandem", inflow=get_les_inflow_windfarm(), solver_overrides=MODEL_KWARGS)
    sol = wf(layout, [(ct, yaw) for ct, yaw in zip(T[:, 2], T[:, 3])])
    wfld = sol.windfield
    wfld.march_to(x=x.max(), y=0.5 * (y.min() + y.max()), z=0.0)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = np.zeros_like(X)
    base = wfld.base_windfield
    u0 = base.wsp(X, Y, Z) * np.cos(base.wdir(X, Y, Z))  # analytic inflow, streamwise component
    du = RegularGridInterpolator((wfld.x, wfld.y, wfld.z), wfld.modules["du"].field, bounds_error=False, fill_value=0.0)
    return u0 + du(np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)).reshape(X.shape), np.array([r.Cp for r in sol.rotors])


def generate(regenerate=False):
    """Dict for the figure: x, y, u_les, u_tandem (nx, ny) [u / U_h], turbines, Cp_tandem (n,), and the LES bookkeeping."""
    les = np.load(HUBPLANE_JOINT)
    out = dict(u_les=les["u"], x=les["x"], y=les["y"], turbines=les["turbines"], **{k: les[k] for k in ("uhub", "tidx", "budget_n", "t_budget_start")})
    if CACHE.exists() and not regenerate:
        model = np.load(CACHE)
        return {**out, "u_tandem": model["u_tandem"], "Cp_tandem": model["Cp_tandem"]}
    u_tandem, Cp = tandem_hubplane(out["turbines"], out["x"], out["y"])
    np.savez_compressed(CACHE, u_tandem=u_tandem, Cp_tandem=Cp)
    return {**out, "u_tandem": u_tandem, "Cp_tandem": Cp}


if __name__ == "__main__":
    if "--les" in sys.argv:
        np.savez_compressed(HUBPLANE_JOINT, **read_les())
    d = generate(regenerate=True)
    print(f"{HUBPLANE_JOINT.name}, {CACHE.name}: grid {d['u_les'].shape}, LES U_h = {float(d['uhub']):.4f}, tidx = {int(d['tidx'])}")
