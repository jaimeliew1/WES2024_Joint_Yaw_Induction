# dualitic is imported first to ensure correct monkey patching.
import dualitic

"""
R1: Welch's t-test (unequal variances) on the LES farm power of the farm-control runs. No figure; prints tables.

The samples are the farm-mean C_P of the 10-minute blocks of each run (generate/generate_les_farm_power.py caches), the raw power and
not the gain over No control, so the No-control uncertainty does not enter. Per comparison: difference of the means (in C_P and in
percentage points of the No-control C_P), its 95% confidence interval from the Welch-Satterthwaite degrees of freedom, t and the
two-sided p. The blocks are treated as independent (lag-1 autocorrelation of the block means is printed to check).

Second table: the same tests with every run cut to the first N blocks, N the fewest blocks of any run (a common time window). The
Yaw K Omega^2 run drifts upward with time, so the comparison with it depends on the window.

Usage: python -m WES2024.R1.forpaper.print_les_farm_welch
"""

import numpy as np
from scipy import stats

from WES2024.R1.generate.generate_les_farm_power import generate

NAMES = {"NoControl": "No control", "Yaw_TANDEM": "Yaw KOmega^2 (TANDEM)", "YawMod_TANDEM": "Yaw lambda~ (TANDEM)",
         "Joint_TANDEM": "Joint (TANDEM)", "Joint_Vortex": "Joint (Vortex sheet)", "Joint_Gauss": "Joint (Gaussian)"}
RUNS = ["Yaw_TANDEM", "YawMod_TANDEM", "Joint_TANDEM", "Joint_Vortex", "Joint_Gauss"]
# (a, b): test of mean(a) - mean(b)
PAIRS = [("YawMod_TANDEM", "Joint_TANDEM"), ("YawMod_TANDEM", "Yaw_TANDEM"), ("Joint_Vortex", "Joint_TANDEM"),
         ("Joint_Vortex", "Yaw_TANDEM"), ("Joint_Vortex", "YawMod_TANDEM"), ("Yaw_TANDEM", "Joint_TANDEM")]


def farm_blocks():
    """{case: farm-mean C_P of every 10-minute block}."""
    return {k: generate(k)["blocks"].mean(axis=0) for k in ["NoControl"] + RUNS}


def welch(x, y):
    """(mean difference, 95% CI half-width, t, dof, p) of mean(x) - mean(y), unequal variances."""
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    dof = se**4 / (vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1))
    t, p = stats.ttest_ind(x, y, equal_var=False)
    return x.mean() - y.mean(), stats.t.ppf(0.975, dof) * se, t, dof, p


def print_tests(b, title):
    nc = b["NoControl"].mean()
    print(f"\n{title}")
    print(f"{'a vs b':50s}{'dCp':>9s}{'(pts of NC)':>13s}{'95% CI +-':>11s}{'t':>7s}{'dof':>6s}{'p':>8s}")
    for a, c in PAIRS:
        d, ci, t, dof, p = welch(b[a], b[c])
        print(f"{NAMES[a] + ' vs ' + NAMES[c]:50s}{d:+9.4f}{100 * d / nc:+13.2f}{ci:11.4f}{t:+7.2f}{dof:6.1f}{p:8.3f}")


def main():
    b = farm_blocks()
    print(f"{'case':24s}{'blocks':>7s}{'mean Cp':>9s}{'SD':>8s}{'SE':>8s}{'lag-1 r':>9s}")
    for k, x in b.items():
        r = np.corrcoef(x[:-1], x[1:])[0, 1]
        print(f"{NAMES[k]:24s}{len(x):7d}{x.mean():9.4f}{x.std(ddof=1):8.4f}{x.std(ddof=1) / np.sqrt(len(x)):8.4f}{r:9.2f}")
    print_tests(b, "Welch's t-test on the farm C_P of the 10-minute blocks, all blocks of each run")
    n = min(len(b[k]) for k in RUNS)
    nc = b["NoControl"]
    b = {k: x[:n] for k, x in b.items()}
    b["NoControl"] = nc  # the normalisation of 'pts of NC' stays the full No-control mean
    print_tests(b, f"Same, every run cut to its first {n} blocks (common time window)")


if __name__ == "__main__":
    main()
