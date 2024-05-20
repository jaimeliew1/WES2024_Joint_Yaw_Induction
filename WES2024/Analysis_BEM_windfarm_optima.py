"""
Analysis...

Key points:
- ??

"""
from itertools import product
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from foreach import foreach
from MITRotor.ReferenceTurbines import IEA15MW
from mitwindfarm import BEM, Layout, Windfarm
from optimise import JointControlBEM

from WES2024 import utils
from WES2024.BEM_gradients import DualBEM

FILESTEM = Path(__file__).stem

REGENERATE = False
PARALLEL = True


windfarm = Windfarm(rotor_model=BEM(IEA15MW(), BEM_model=DualBEM))

TURBINE_SPACING = 6.0
layouts = {
    "2_turb": Layout([TURBINE_SPACING * i for i in range(2)], [0.0, 0.0]),
    "3_turb": Layout([TURBINE_SPACING * i for i in range(3)], [0.0, 0.0, 0.0]),
    "4_turb": Layout([TURBINE_SPACING * i for i in range(4)], [0.0, 0.0, 0.0, 0.0]),
    "5_turb": Layout([TURBINE_SPACING * i for i in range(5)], [0.0, 0.0, 0.0, 0.0, 0.0]),
}
wdirs = np.arange(-90.0, 90.0, 0.1)


methods = {
    "JointControl": JointControlBEM,
}


def _generate(x):
    method, wdir, layout_name = x
    sol = methods[method](layouts[layout_name].rotate(wdir), windfarm).optimise(
        Cp_constraint=None,
        use_gradients=True,
        verbose=False,
    )

    return utils.to_polars(sol).with_columns(
        pl.lit(method).alias("method"),
        pl.lit(wdir).alias("wdir"),
        pl.lit(layout_name).alias("layout"),
    )


@utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False):
    params = list(product(methods, wdirs, layouts.keys()))

    dfs = foreach(_generate, params, parallel=PARALLEL)
    df = pl.concat(dfs)
    return df


def plot(df: pl.DataFrame, layout: str):
    fig = plt.figure()

    df = (
        df.rename(dict(setpoint_0="pitch", setpoint_1="tsr"))
        .with_columns(
            np.rad2deg(pl.col("yaw")),
            np.rad2deg(pl.col("pitch")),
            np.rad2deg(np.abs(pl.col("yaw"))).alias("abs_yaw"),
        )
    )
    if layout is not None:
        df = df.filter(pl.col("layout") == layout)

    norm = mpl.colors.Normalize(0, 40)
    graph = sns.scatterplot(
        df.to_pandas(),
        x="pitch",
        y="tsr",
        hue="abs_yaw",
        hue_norm=norm,
        palette="magma",
        legend=False,
        s=7,
        edgecolors=None,
        ax=plt.gca(),
    )

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="magma"),
        ax=plt.gca(),
        orientation="vertical",
        label=r"$|\gamma|$ [deg]",
    )
    graph.set_xlabel(r"$\theta_p$ [deg]")
    graph.set_ylabel(r"$\lambda$ [-]")

    plt.savefig(utils.FIGDIR / f"{FILESTEM}_{layout}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=REGENERATE)
    plot(df, layout="2_turb")
    plot(df, layout="3_turb")
    plot(df, layout="4_turb")
    plot(df, layout="5_turb")
    plot(df, layout=None)


if __name__ == "__main__":
    main()
