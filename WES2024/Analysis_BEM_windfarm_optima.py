"""
Analysis...

Key points:
- ??

"""
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from foreach import foreach
from MITRotor.ReferenceTurbines import IEA15MW
from mitwindfarm.Layout import Layout
from mitwindfarm.Rotor import BEM
from mitwindfarm.windfarm import Windfarm
from optimise import JointControlBEM, NoControlBEM, ThrustControlBEM, YawControlBEM
from test_BEM_gradients import DualBEM

from WES2024 import utils

FILESTEM = Path(__file__).stem

REGENERATE = False
PARALLEL = True


windfarm = Windfarm(rotor_model=BEM(IEA15MW(), BEM_model=DualBEM))

TURBINE_SPACING = 5.0
layouts = {
    "2_turb": Layout([TURBINE_SPACING * i for i in range(2)], [0.0, 0.0]),
    "3_turb": Layout([TURBINE_SPACING * i for i in range(3)], [0.0, 0.0, 0.0]),
    "4_turb": Layout([TURBINE_SPACING * i for i in range(4)], [0.0, 0.0, 0.0, 0.0]),
    "5_turb": Layout([TURBINE_SPACING * i for i in range(5)], [0.0, 0.0, 0.0, 0.0, 0.0]),
}
wdirs = np.arange(-90.0, 90.0, 0.05)


methods = {
    "NoControl": NoControlBEM,
    # "YawControl": YawControlBEM,
    "ThrustControl": ThrustControlBEM,
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
        pl.lit(method).alias("method"), pl.lit(wdir).alias("wdir")
    )


@utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False):
    params = list(product(methods, wdirs, layouts.keys()))

    dfs = foreach(_generate, params, parallel=PARALLEL)
    df = pl.concat(dfs)
    return df


def plot(df: pl.DataFrame):
    plt.figure()

    df = (
        df.rename(dict(setpoint_0="pitch", setpoint_1="tsr")).filter(
            pl.col("method").is_in(["JointControl"])
        )
        # .filter(pl.col("method").is_in(["ThrustControl"]))
        # .filter(pl.col("method").is_in(["ThrustControl", "JointControl"]))
        .with_columns(
            np.rad2deg(pl.col("yaw")),
            np.rad2deg(pl.col("pitch")),
        )  # .filter(pl.col("group") == "OLE2")
    )

    graph = sns.jointplot(
        df.to_pandas(),
        x="pitch",
        y="tsr",
        # y="Ctprime",
        # x="yaw",
        hue="turbine",
        ratio=3,
        height=4,
        palette="tab10",
        legend=True,
        s=7,
        edgecolors=None,
    )
    sns.move_legend(
        graph.ax_joint,
        "lower center",
        bbox_to_anchor=(0.5, 0.9),
        ncol=4,
        frameon=True,
        fontsize="xx-small",
        title=None,
    )

    graph.ax_joint.set_xlabel(r"$\theta_p$ [deg]")
    graph.ax_joint.set_ylabel(r"$\lambda$ [-]")
    plt.savefig(utils.FIGDIR / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=REGENERATE)
    plot(df)


if __name__ == "__main__":
    main()
