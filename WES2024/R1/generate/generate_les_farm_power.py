# dualitic is imported first to ensure correct monkey patching.
import dualitic

"""
R1: time series of the turbine power of the 5x5 farm LES runs (farm_control_cases.CASES), as 10-minute block means, cached.

Per run: the power of every turbine from the PadeOps ADM `*.pow` files (padeopsIO read_turb_power, cropped to after the budget start
time `time_budget_start` of the input file), normalised by 1/2 U_h^3 pi/4 (D = 1, rho = 1), U_h the hub-height speed of the run's
own precursor (Run04; sqrt(ubar^2 + vbar^2) of the xy average at z_hub); then averaged over non-overlapping 10-minute blocks
(D = 100 m, G = 10 m/s: one time unit = 10 s). Stored in LES_DATA/R1/les_farm_power_<key>.npz with
    blocks (n_turbine, n_blocks) block-mean Cp, row, column (n_turbine,), x, y [D], n_samples, t_span [time units], post_transient.

Statistics (first blocked, then averaged, so the covariance between turbines inside a block is kept and the standard error is that of
the 10-minute blocks, `*_stats` functions below):
    les_farm_stats(key)   mean and SE of the farm-mean Cp
    les_row_stats(key)    mean and SE of the row-mean Cp (5,)
    ratio_with_se(a, sa, b, sb)   a / b with first-order propagated SE for two independent runs.
Normalisations in the figures: LES curves are divided by the square_NoControl row-1 mean (rows) or farm mean (gain).

`python -m WES2024.R1.generate.generate_les_farm_power` re-reads every run from the raw LES (padeopsIO, /scratch); the figures only read LES_DATA/R1.
"""

import numpy as np
from scipy.interpolate import interp1d

from WES2024.R1.farm_control_cases import CASES, SPACING, les_path
from WES2024.R1.les_data import farm_power_path

TO_SEC = 100.0 / 10.0  # D / G [s per time unit]
BLOCK_SECONDS = 600


def _time_axis(b):
    """Time of every step of the run, linearly interpolated between the written Time IDs (padeopsIO unique_times / unique_tidx),
    extended back to the restart step if the run was restarted."""
    import padeopsIO as pio

    times, tidx = b.unique_times(), b.unique_tidx()
    first = tidx[0]
    if pio.key_search_r(b.input_nml, "userestartfile"):
        first = min(pio.key_search_r(b.input_nml, "restartfile_tid"), tidx[0])
    new_tidx = np.arange(first, np.max(tidx) + 1, 1)
    return interp1d(tidx, times, kind="linear", fill_value="extrapolate")(new_tidx)


def _powers(b, norm):
    """(time [time units] after the budget start, list of n_turbine power series / norm)."""
    import padeopsIO as pio

    t_start = pio.key_search_r(b.input_nml, "time_budget_start")
    t = _time_axis(b)
    keep = (t > t_start) if np.any(t > t_start) else np.ones(len(t), bool)
    series = [np.asarray(b.read_turb_power(turb=k, tidx="all", steady=False)) / norm for k in range(1, b.n_turb + 1)]
    n = min(len(s) for s in series)  # the series can differ in length by a step
    try:
        series = [s[-n:][keep] for s in series]
    except IndexError:
        series = [s[-keep.sum():] for s in series]
    if np.isnan(series).any():
        raise ValueError("nan in turbine power")
    return t[keep], series


def read_les(path):
    """Dict of the arrays cached (see module docstring) read straight from the LES directory `path`."""
    import padeopsIO as pio

    c = pio.BudgetIO(str(path), padeops=True, runid=5, normalize_origin=None)
    p = pio.BudgetIO(c.dirname, padeops=True, runid=4, normalize_origin=c.origin)
    if not (c.associate_budgets and p.associate_budgets):
        raise RuntimeError(f"budgets not associated for {path}")
    s = p.xy_avg(budget_terms=["ubar", "vbar"])
    u_h = float(np.interp(1.0, s["z"], np.sqrt(s["ubar"] ** 2 + s["vbar"] ** 2)))  # z_hub = 1 in the un-normalised frame
    t, power = _powers(c, 0.5 * u_h**3 * np.pi / 4)
    power = np.asarray(power)
    win = max(int(BLOCK_SECONDS / (float(np.mean(np.diff(t))) * TO_SEC)), 1)
    n_b = power.shape[1] // win
    blocks = power[:, : n_b * win].reshape(len(power), n_b, win).mean(axis=2)

    c.ta.sort(by="n")
    x = np.array([adm.xloc for adm in c.ta])
    y = np.array([adm.yloc for adm in c.ta])
    x, y = x - x.min(), y - y.min()
    return dict(blocks=blocks, x=x, y=y, row=np.round(x / SPACING).astype(int), column=np.round(y / SPACING).astype(int),
                n_samples=power.shape[1], t_span=float(np.ptp(t)), uhub=u_h,
                post_transient=bool(np.min(t) > pio.key_search_r(c.input_nml, "time_budget_start")))


def generate(key, regenerate=False):
    """Stored dict for the case `key` of farm_control_cases.CASES (read from /scratch only if it is missing or regenerate)."""
    f = farm_power_path(key)
    if f.exists() and not regenerate:
        d = np.load(f)
        return {k: d[k] for k in d}
    return {k: np.asarray(v) for k, v in read_les(les_path(key)).items()}


# ---------------------------------------------------------------------------------------------------------------------------------
def _mean_se(blocks):
    n = blocks.shape[-1]
    se = blocks.std(axis=-1, ddof=1) / np.sqrt(n) if n > 1 else np.full(blocks.shape[:-1], np.nan)
    return blocks.mean(axis=-1), se


def les_farm_stats(key):
    """(mean Cp_farm, SE): turbine-average per block first, then SE over the 10-minute blocks."""
    m, se = _mean_se(generate(key)["blocks"].mean(axis=0))
    return float(m), float(se)


def _row_blocks(d):
    return np.stack([d["blocks"][d["row"] == r].mean(axis=0) for r in range(5)])


def les_row_stats(key):
    """(mean Cp per row (5,), SE (5,)), averaged over all columns."""
    return _mean_se(_row_blocks(generate(key)))


def les_n_blocks(key):
    return generate(key)["blocks"].shape[1]


def ratio_with_se(a, sa, b, sb):
    """a / b with first-order propagated SE, a and b from independent runs."""
    r = np.asarray(a) / b
    return r, np.abs(r) * np.sqrt((np.asarray(sa) / np.asarray(a)) ** 2 + (sb / b) ** 2)


def les_row_ratio_to_row0(key):
    """(mean, SE) of <P>_row / <P>_row0 of the SAME run: the ratio is formed per block before averaging (row 0 -> exactly 1)."""
    rows = _row_blocks(generate(key))
    return _mean_se(rows / rows[0])


if __name__ == "__main__":
    for key in CASES:
        d = generate(key, regenerate=True)
        np.savez_compressed(farm_power_path(key), **d)
        m, se = _mean_se(d["blocks"].mean(axis=0))
        print(f"{key:14s} {d['blocks'].shape[1]:3d} blocks, t_span {float(d['t_span']):7.1f}, farm Cp {m:.4f} +- {se:.4f} -> {farm_power_path(key).name}")
