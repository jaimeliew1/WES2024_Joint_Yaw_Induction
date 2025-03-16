"""
Figure 3:
2 turbine wind direction sweep

This figure shows the optimal setpoints (Cp, yaw, pitch, TSR, Ct' ) using four
different control strategies (None, yaw, thrust, and joint control) and two
different modelling methods (AD and BEM).

Key points:
- Joint control performs better than all other methods.
- AD and BEM show good agreement in setpoints. Losses exist in BEM.
- Yaw control is discontinuous.
- Joint control is smooth.
- Yaw control has non-constant thrust. i.e. yaw and thrust are coupled quantities.

"""
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from WES2024 import utils
from WES2024.Generate import two_turbine_AD, two_turbine_BEM

REGENERATE = False
FILESTEM = Path(__file__).stem

INCLUDE_K_OMEGA = False

plot_params = {
    ("NoControl", "AD"): dict(
        label=f"{utils.controller_labels['NoControl']} (AD)",
        c="0.0",
    ),
    ("NoControl", "BEM"): dict(
        label=f"{utils.controller_labels['NoControl']} (BEM)",
        ls="--",
        c="0.6",
    ),
    ("ThrustControl", "AD"): dict(
        label=f"{utils.controller_labels['ThrustControl']} (AD)",
        c=plt.cm.tab20(0 / 20),
    ),
    ("ThrustControl", "BEM"): dict(
        label=f"{utils.controller_labels['ThrustControl']} (BEM)",
        ls="--",
        c=plt.cm.tab20((0 + 1) / 20),
    ),
    ("YawControl", "AD"): dict(
        label=f"{utils.controller_labels['YawControl']} (AD)",
        c=plt.cm.tab20(2 / 20),
    ),
    ("YawControl", "BEM"): dict(
        label=f"{utils.controller_labels['YawControl']} (BEM)",
        ls="--",
        c=plt.cm.tab20((2 + 1) / 20),
    ),
    ("JointControl", "AD"): dict(
        label=f"{utils.controller_labels['JointControl']} (AD)",
        c=plt.cm.tab20(4 / 20),
    ),
    ("JointControl", "BEM"): dict(
        label=f"{utils.controller_labels['JointControl']} (BEM)",
        ls="--",
        c=plt.cm.tab20((4 + 1) / 20),
    ),
    ("YawKOmegaControl", "BEM"): dict(
        label=f"{utils.controller_labels['YawKOmegaControl']} (BEM)",
        ls=":",
        c=plt.cm.tab20((6 + 1) / 20),
    ),
}


axis_params = {
    "Cp_AD": dict(
        ylabel=r"$C_P$",
        title="a) AD power coefficient",
        ylim=(0.35, 0.65),
    ),
    "Cp_BEM": dict(
        ylabel=r"$C_P$",
        title="b) BEM power coefficient",
        ylim=(0.35, 0.65),
    ),
    "Cp": dict(
        ylabel=r"$C_P$",
        title="a) power coefficient",
        ylim=(0.35, 0.65),
    ),
    "Ctprime": dict(
        ylabel=r"$C_T'$",
        title="d) Optimal modified thrust coefficient",
        ylim=(1.0, 2.5),
    ),
    "Ct": dict(
        ylabel=r"$C_T$",
        title="b) Optimal thrust coefficient",
        ylim=(0.6, 1.0),
    ),
    "yaw": dict(
        ylabel=r"$\gamma$ (deg)",
        title="c) Optimal yaw angle",
        ylim=(-30, 30),
    ),
    "pitch": dict(
        ylabel=r"$\theta_p$ (deg)",
        title="e) Optimal blade pitch",
        ylim=(-1.5, 2.0),
    ),
    "tsr": dict(
        ylabel=r"$\lambda$",
        title="f) Optimal tip speed ratio",
        ylim=(8.0, 9.5),
    ),
}


def generate(regenerate=False):
    # Load AD data. Convert angles to degrees.
    df_AD = (
        # two_turbine_AD.generate(regenerate=regenerate)
        two_turbine_AD.generate(regenerate=regenerate)
        .select(pl.exclude("setpoint_0", "setpoint_1"))
        .with_columns(
            pl.col("yaw").degrees(),
            pl.col("Cp").alias("Cp"),
            pl.lit("AD").alias("type"),
        )
    )
    # Load BEM data. Convert angles to degrees and rename setpoints to pitch and
    # tsr.
    df_BEM = (
        two_turbine_BEM.generate(regenerate=regenerate)
        .select(pl.exclude("setpoint_2"))
        .with_columns(
            pl.col("setpoint_0").degrees().alias("pitch"),
            pl.col("setpoint_1").alias("tsr"),
            pl.col("yaw").degrees(),
            pl.col("Cp").alias("Cp"),
            pl.lit("BEM").alias("type"),
        )
    )
    # Concatenate AD and BEM data. remove yaw control data points for wdir=0 to
    # highlight discontinuity.
    df = pl.concat([df_AD, df_BEM], how="diagonal_relaxed").with_columns(
        pl.when(
            pl.col("method").is_in(["YawControl", "YawKOmegaControl"]), pl.col("wdir").abs() < 1e-1
        )
        .then(np.nan)
        .otherwise(pl.col("yaw"))
        .alias("yaw"),
        pl.when(
            pl.col("method").is_in(["YawControl", "YawKOmegaControl"]), pl.col("wdir").abs() < 1e-1
        )
        .then(np.nan)
        .otherwise(pl.col("tsr"))
        .alias("tsr"),
        pl.when(
            pl.col("method").is_in(["YawControl", "YawKOmegaControl"]), pl.col("wdir").abs() < 1e-1
        )
        .then(np.nan)
        .otherwise(pl.col("Cp"))
        .alias("Cp"),
    )
    return df


def plot(df):
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=2 * np.array([5, 3]))
    plt.subplots_adjust(wspace=0.2)

    keys = ["Cp", "Ct", "yaw", "Ctprime", "pitch", "tsr"]
    methods = ["NoControl", "ThrustControl", "YawControl", "JointControl"]
    if INCLUDE_K_OMEGA:
        methods += ["YawKOmegaControl"]
    sim_types = ["AD", "BEM"]
    for (ax, key), method, sim_type in product(zip(axes.ravel(), keys), methods, sim_types):
        if (method, sim_type) == ("YawKOmegaControl", "AD"):
            continue
        _df = (
            df.filter(pl.col("method") == method)
            .filter(pl.col("type") == sim_type)
            .group_by("wdir")
            .agg(
                pl.col("Cp").mean(),
                # pl.col("Cp").mean(),
                pl.col("Ctprime").where(pl.col("turbine") == 0).first(),
                pl.col("Ct").where(pl.col("turbine") == 0).first(),
                pl.col("yaw").where(pl.col("turbine") == 0).first(),
                pl.col("pitch").where(pl.col("turbine") == 0).first(),
                pl.col("tsr").where(pl.col("turbine") == 0).first(),
            )
            .sort("wdir")
        )

        ax.plot(_df["wdir"], _df[key], **plot_params[method, sim_type])
    plt.xlim(-20, 20)

    # Apply various axis parameters
    for ax, key in zip(axes.ravel(), keys):
        # Set ylabels
        ax.set_ylabel(axis_params[key]["ylabel"])
        ax.text(0.01, 0.99, axis_params[key]["title"], ha="left", va="top", transform=ax.transAxes)
        # Set same ylimits
        if axis_params[key]["ylim"]:
            ax.set_ylim(axis_params[key]["ylim"])

    # Set xlabels
    axes[2, 0].set_xlabel("wind direction (deg)")
    axes[2, 1].set_xlabel("wind direction (deg)")

    # Legend
    axes[0, 0].legend(loc="lower center", ncol=4, bbox_to_anchor=(1.1, 1.01))

    plt.savefig(utils.FIGDIRFORPAPER / f"{FILESTEM}.png", dpi=300, bbox_inches="tight")


def main():
    df = generate(regenerate=REGENERATE)
    plot(df)


if __name__ == "__main__":
    main()
