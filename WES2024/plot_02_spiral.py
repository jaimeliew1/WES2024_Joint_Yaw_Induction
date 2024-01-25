from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from cache import cache_polars
from foreach import foreach
from mitwindfarm import Plotting
from mitwindfarm.Layout import Layout
from mitwindfarm.windfarm import Windfarm
from optimise import JointControl, NoControl, ThrustControl, YawControl
from profilehooks import profile
from scipy.spatial import distance
from utilities import from_polars, to_polars


# Use Latex Fonts
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})



FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)

windfarm = Windfarm()


def generate_sunflower_spiral(N: int, min_dist: float):
    idx = np.arange(0, N) + 0.5
    r = np.sqrt(idx / N)
    theta = np.pi * (1 + np.sqrt(5)) * idx

    X, Y = r * np.cos(theta), r * np.sin(theta)
    dist_min = distance.pdist(np.array(list(zip(X, Y)))).min()
    X *= min_dist / dist_min
    Y *= min_dist / dist_min
    return X, Y


class Spiral(Layout):
    def __init__(self, N: int, min_dist: float):
        X, Y = generate_sunflower_spiral(N, min_dist)

        super().__init__(X, Y)


methods = {
    "NoControl": NoControl,
    "YawControl": YawControl,
    "ThrustControl": ThrustControl,
    "JointControl": JointControl,
}

plot_params = {
    "NoControl": dict(ls="--", c="k", label="No Control"),
    "ThrustControl": dict(c="tab:blue", label="Thrust Control"),
    "YawControl": dict(c="tab:orange", label="Yaw Control"),
    "JointControl": dict(c="tab:green", label="Joint Control"),
}

layouts = {
    4: Spiral(20, min_dist=4),
    6: Spiral(20, min_dist=6),
    8: Spiral(20, min_dist=8),
    10: Spiral(20, min_dist=10),
}
wdirs = np.arange(0, 360, 1)


# @profile(filename="prof.prof")
def _generate(x):
    method, wdir, min_dist = x
    sol = methods[method](layouts[min_dist].rotate(wdir), windfarm).optimise(use_gradients=True)
    return to_polars(sol).with_columns(
        pl.lit(method).alias("method"),
        pl.lit(wdir).alias("wdir"),
        pl.lit(min_dist).alias("min_dist"),
    )


@cache_polars(Path(__file__).parent.parent / "data/plot_02_spiral.csv")
def generate(regenerate=False):
    params = list(product(methods, wdirs, layouts))

    df = pl.concat(foreach(_generate, params, parallel=True))
    return df


def plot_Cp_vs_distance(df: pl.DataFrame):
    N_dist = df["min_dist"].n_unique()

    fig, axes = plt.subplots(N_dist, 1, sharex=True, sharey=True)

    for ax, min_dist in zip(axes, df["min_dist"].unique().sort()):
        ax.text(0.5, 0.99, f"turbine spacing: {min_dist}D", ha="center", va="top", transform=ax.transAxes)
        for method, _plot_params in plot_params.items():
            _df = df.filter(pl.col("min_dist") == min_dist).filter(pl.col("method") == method)
            to_plot = _df.group_by("wdir").agg(pl.col("Cp").mean()).sort("wdir")
            ax.plot(to_plot["wdir"], to_plot["Cp"], **plot_params[method])

    axes[-1].set_xlabel("wind direction (deg)")
    [ax.set_ylabel("$C_P$") for ax in axes]

    axes[0].set_ylim(0.3, 0.65)

    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4)

    axes[0].set_xlim(0, 360)
    plt.savefig(FIGDIR / "spiral_Cp_vs_distance.png", dpi=300, bbox_inches="tight")


def plot_farm_performance_vs_distance(df: pl.DataFrame):
    df_farm_Cp = df.pivot(columns="method", index="min_dist", values="Cp", aggregate_function="mean")
    df_farm_Cp = df_farm_Cp.select(
        pl.col("min_dist"),
        pl.exclude("min_dist", "NoControl") / pl.col("NoControl") * 100 - 100,
    )
    print(df_farm_Cp)

    methods = df_farm_Cp.columns
    methods.remove("min_dist")

    plt.figure()
    for method, _plot_params in plot_params.items():
        if method == "NoControl":
            continue
        plt.plot(df_farm_Cp["min_dist"], df_farm_Cp[method], **_plot_params)

    plt.legend()

    plt.xlabel("Turbine spacing [D]")
    plt.ylabel(r"Power increase [\%]")
    plt.savefig(FIGDIR / "spiral_farm_performance_vs_distance.png", dpi=300, bbox_inches="tight")


def plot_windfarm(df: pl.DataFrame):
    for min_dist in df["min_dist"].unique():
        _df = (
            df.filter(pl.col("method") == "JointControl")
            .filter(pl.col("min_dist") == min_dist)
            .filter(pl.col("wdir") == 0)
        )
        windfarm_sol = from_polars(_df, windfarm)

        Plotting.plot_windfarm(windfarm_sol)
        plt.savefig(FIGDIR / f"spiral_windfarm_{min_dist}D.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_setpoints(df: pl.DataFrame):
    df = df.filter(pl.col("min_dist") == 4).filter(pl.col("method") == "JointControl")

    fig, axes = plt.subplots(2, 1, sharex=True)

    for i, _df in df.group_by("turbine"):
        axes[0].plot(_df["wdir"], np.rad2deg(_df["yaw"]), c=plt.cm.gist_ncar(i / 20))
        axes[1].plot(_df["wdir"], _df["Ctprime"], c=plt.cm.gist_ncar(i / 20))

    axes[-1].set_xlabel("wind direction [deg]")

    axes[0].set_ylabel("yaw setpoint [deg]")
    axes[1].set_ylabel("$C_T'$ setpoint")

    axes[0].set_xlim(0, 360)
    plt.savefig(FIGDIR / "spiral_setpoints.png", dpi=300, bbox_inches="tight")


def plot_POD(df: pl.DataFrame):
    df = df.filter(pl.col("min_dist") == 4).filter(pl.col("method") == "JointControl")
    yaws = np.rad2deg(df.pivot(index="turbine", columns="wdir", values="yaw").to_numpy())
    yaws = df.pivot(index="turbine", columns="wdir", values="yaw").to_numpy()
    ctprimes = df.pivot(index="turbine", columns="wdir", values="Ctprime").to_numpy()
    setpoints = np.vstack([yaws, ctprimes])[:, :-1]  # there is an extra point at the end.

    mean = setpoints.mean(axis=1)
    setpoints -= mean[:, np.newaxis]
    U, S, V = np.linalg.svd(setpoints)

    plt.figure()
    plt.plot(S / S.sum())

    plt.savefig(FIGDIR / "spiral_POD_S.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    # plt.plot(U[:,0:5], ".") # first mode shape (?)
    # plt.plot(U[:,0], ".") # first mode shape (?)
    plt.plot(V[1:5, :].T)  # Mode magnitude over yaw angles
    # plt.plot(mean)
    plt.savefig(FIGDIR / "spiral_POD.png", dpi=300, bbox_inches="tight")
    plt.close()
    # breakpoint()


def main():
    df = generate(regenerate=False)
    plot_POD(df)
    plot_setpoints(df)
    plot_windfarm(df)
    plot_farm_performance_vs_distance(df)
    plot_Cp_vs_distance(df)


if __name__ == "__main__":
    main()
