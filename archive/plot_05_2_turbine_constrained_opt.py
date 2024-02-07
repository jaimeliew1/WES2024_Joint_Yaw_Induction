from pathlib import Path
from itertools import product

import matplotlib.pyplot as plt
from foreach import foreach
import numpy as np
import polars as pl
from cache import cache_polars
from mitwindfarm.Layout import Layout
from mitwindfarm.windfarm import Windfarm
from utilities import from_polars, to_polars
from optimise import JointControl, NoControl, ThrustControl, YawControl

# Use Latex Fonts
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)

windfarm = Windfarm()
layout = Layout([0, 7], [0.0, 0.0])
wdirs = np.arange(-20, 20, 0.25)
Cp_constraints = np.arange(0.4, 0.6, 0.02)


methods = {
    # "NoControl": NoControl,
    # "YawControl": YawControl,
    "ThrustControl": ThrustControl,
    "JointControl": JointControl,
}


def _generate(x) -> pl.DataFrame:
    method, wdir, Cp_constraint = x
    sol = methods[method](layout.rotate(wdir), windfarm).optimise(Cp_constraint=Cp_constraint)
    return to_polars(sol).with_columns(method=pl.lit(method), wdir=wdir, Cp_constraint=Cp_constraint)


@cache_polars(Path(__file__).parent.parent / "data/plot_05_2_turbine_constrained_opt.csv")
def generate(regenerate=False) -> pl.DataFrame:
    params = list(product(methods, wdirs, Cp_constraints))

    df = pl.concat(foreach(_generate, params, parallel=True))
    return df


def plot(df):
    fig, axes = plt.subplots(3, 1, sharex=True)

    for method in methods:
        _df = (
            df.filter(pl.col("method") == method)
            .group_by("wdir")
            .agg(pl.col("Cp").mean(), pl.col("yaw").first(), pl.col("Ctprime").first())
        ).sort("wdir")

        axes[0].plot(_df["wdir"], _df["Cp"], label=method)
        axes[1].plot(_df["wdir"], _df["Ctprime"], label=method)
        axes[2].plot(_df["wdir"], np.rad2deg(_df["yaw"]), label=method)

    axes[-1].set_xlabel("wind direction [deg]")

    axes[0].set_ylabel("$C_P$")
    axes[1].set_ylabel("$C_T'$")
    axes[2].set_ylabel(r"$\gamma$ [deg]")

    axes[0].set_ylim(0.3, 0.60)
    axes[0].legend()

    plt.savefig(FIGDIR / "2_turbine_constrained_opt.png", dpi=300, bbox_inches="tight")
    # This figure is incomplete! (data processing is wrong)


if __name__ == "__main__":
    df = generate(regenerate=True)

    plot(df)
    print(df)

    pl.Config.set_tbl_rows(100)
    print(df.pivot(index="Cp_constraint", columns="method", values="Cp", aggregate_function="mean"))
