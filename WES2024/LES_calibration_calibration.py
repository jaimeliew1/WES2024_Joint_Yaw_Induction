from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import polars as pl
from mitwindfarm import (
    GaussianWake,
    RotorSolution,
    WakeModel,
    Windfarm,
    WindfarmSolution,
)
from mitwindfarm.Layout import Layout
from rich import print
from scipy.optimize import minimize
from tqdm import tqdm

from WES2024.Generate import LES_case_definitions

LES_FN = Path("data/mean_power_wdir-2.5.csv")
row_indices = [
    [24],
    [23, 19],
    [22, 18, 14],
    [21, 17, 13, 9],
    [20, 16, 12, 8, 4],
    [15, 11, 7, 3],
    [10, 6, 2],
    [5, 1],
    [0],
]

df = (
    LES_case_definitions.generate()
    .filter(wdir=-2.5, controller="nocontrol")
    .select("turbine", "x", "y")
)
df = df

N_TURB = 25
LAYOUT = Layout(df["x"].to_numpy(), df["y"].to_numpy())
setpoints = len(LAYOUT) * [(2.0, 0.0)]
SIGMA = [1 / np.sqrt(8) for _ in range(N_TURB)]
KW_INIT = 0.07


class IndividuallyCalibratedGaussianWakeModel(WakeModel):
    def __init__(self, sigma: list, kw: list):
        self.sigma = sigma
        self.kw = kw

    def __call__(self, x, y, z, rotor_sol: "RotorSolution") -> GaussianWake:
        idx = rotor_sol.idx
        return GaussianWake(x, y, z, rotor_sol, sigma=self.sigma[idx], kw=self.kw[idx])


def run(sigma: list[float], kw: list[float]) -> WindfarmSolution:
    wakemodel = IndividuallyCalibratedGaussianWakeModel(sigma=sigma, kw=kw)

    windfarm = Windfarm(wake_model=wakemodel)
    sol = windfarm(LAYOUT, setpoints)
    return sol


def calibrate_row(
    upstream: list[int], downstream: list[int], Cp_downstream: list[float], kw0: list[float]
) -> list[float]:
    """
    Callibrate the wake of the upstream turbines (upstream) based on the Cp
    (Cp_downstream) of the downstream turbines (downstream).
    """
    assert len(upstream) == len(downstream)
    assert len(Cp_downstream) == len(downstream)

    kw = np.array(kw0)

    def func(x):
        for i, idx in enumerate(upstream):
            kw[idx] = x[i]
        sol = run(SIGMA, kw)
        Cp_model = np.array([sol.rotors[idx].Cp for idx in downstream])

        cost = np.sum((Cp_model - Cp_downstream) ** 2)
        return cost

    opt_sol = minimize(func, [KW_INIT for _ in upstream], bounds=[(0, 1) for _ in upstream])

    print("iteration converged :)" if opt_sol.success else "iteration not converged :(")

    for i, idx in enumerate(upstream):
        kw[idx] = opt_sol.x[i]

    return kw


def plot_text_on_layout(layout: Layout, vals: list, fn: Path, title=None):
    plt.figure()
    plt.axis("equal")

    cmap = plt.cm.viridis
    norm = colors.Normalize(vmin=np.min(vals), vmax=np.max(vals))
    for idx, (x, y, val) in enumerate(zip(layout.x, layout.y, vals)):
        plt.plot(x, y, ".", ms=10, c=cmap(norm(val)))
        plt.text(x, y, f"{val:2.3f}")

    if title:
        plt.title(title)

    plt.savefig(fn, dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    df_LES = pl.read_csv(LES_FN)
    print(df_LES)
    kw = [KW_INIT for _ in range(N_TURB)]

    # first (n-1)row to nth row
    for row_num in tqdm([1, 2, 3, 4]):
        upstream, downstream = [], []
        for row in row_indices:
            if len(row) < row_num + 1:
                continue
            upstream.append(row[row_num - 1])
            downstream.append(row[row_num])

        Cp_downstream = df_LES.filter(pl.col("turbine_id").is_in(downstream))

        kw = calibrate_row(upstream, downstream, Cp_downstream["Cp"].to_numpy(), kw)

    plot_text_on_layout(LAYOUT, kw, "kw_on_layout.png", title=r"calibrated $k_w$ map")
