from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from cache import cache_polars
from foreach import foreach
from mitwindfarm.Layout import Layout
from mitwindfarm.windfarm import Windfarm
from optimise import JointControl, NoControl, ThrustControl, YawControl

from utilities import to_polars

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)

windfarm = Windfarm()
layout = Layout([0, 7], [0.0, 0.0])

sigmas = np.arange(11.0)

methods = {
    "NoControl": NoControl,
    "YawControl": YawControl,
    "ThrustControl": ThrustControl,
    "JointControl": JointControl,
}


def _generate(x):
    method, wdir = x
    sol = methods[method](layout.rotate(wdir), windfarm).optimise()

    return to_polars(sol).with_columns(
        pl.lit(method).alias("method"), pl.lit(wdir).alias("wdir")
    )


@cache_polars(
    Path(__file__).parent.parent / "data/plot_03_2_turbine_uncertain_wdir_opt.csv"
)
def generate_opt(regenerate=False):
    wdirs = np.arange(-20, 20, 0.25)
    params = list(product(methods, wdirs))

    df = pl.concat(foreach(_generate, params, parallel=False))
    return df


def extract_layout_and_setpoints(partial: pl.DataFrame) -> (Layout, list[tuple]):
    x, y, z = [], [], []
    setpoints = []
    for _df in partial.iter_rows(named=True):

        x.append(_df["x"])
        y.append(_df["y"])
        z.append(_df["z"])
        setpoints.append(tuple(v for k, v in _df.items() if k.startswith("setpoint_")))
    return Layout(x, y, z), setpoints


@cache_polars(
    Path(__file__).parent.parent / "data/plot_03_2_turbine_uncertain_wdir.csv"
)
def generate(df_opt: pl.DataFrame, regenerate=False) -> pl.DataFrame:
    out = []
    for sigma in sigmas:
        dwdir = np.linspace(-sigma, sigma, 10)

        for (wdir, method), _df in df_opt.group_by(["wdir", "method"]):
            layout, setpoints = extract_layout_and_setpoints(_df)

            for _dwdir in dwdir:
                asdf = to_polars(
                    windfarm(layout.rotate(_dwdir), setpoints)
                ).with_columns(
                    wdir=wdir, eps=_dwdir, sigma=sigma, method=pl.lit(method)
                )
                out.append(asdf)

    return pl.concat(out)


def plot(df: pl.DataFrame):
    # N_sigma = df["sigma"].n_unique()
    # sigmas = df["sigma"].unique().sort()
    sigmas = [0, 3, 6, 9]
    sigmas = [0, 5, 10]
    N_sigma = len(sigmas)
    fig, axes = plt.subplots(N_sigma, 1, sharex=True, sharey=True)

    for ax, sigma in zip(axes, sigmas):
        for method in methods:
            _df = (
                df.filter(pl.col("method") == method)
                .filter(pl.col("sigma") == sigma)
                .group_by("wdir")
                .agg(
                    pl.col("Cp").mean(),
                    pl.col("yaw").first(),
                    pl.col("Ctprime").first(),
                )
            ).sort("wdir")
            ax.plot(_df["wdir"], _df["Cp"], label=method)

            ax.text(
                0.5,
                0.99,
                rf"$\sigma={sigma}^o$",
                ha="center",
                va="top",
                transform=ax.transAxes,
            )

    axes[-1].set_xlabel("wind direction [deg]")

    [ax.set_ylabel("$C_P$") for ax in axes]

    axes[0].set_ylim(0.4, 0.60)
    axes[0].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01))

    plt.savefig(FIGDIR / "2_turbine_uncertain_wdir.png", dpi=300, bbox_inches="tight")


def main():
    df_opt = generate_opt(regenerate=False)
    df = generate(df_opt, regenerate=False)

    df_piv = df.pivot(
        index="sigma", columns="method", values="Cp", aggregate_function="mean"
    )
    df_farm_Cp = df_piv.select(
        pl.col("sigma"),
        pl.exclude("sigma", "NoControl") / pl.col("NoControl") * 100 - 100,
    )
    pl.Config.set_tbl_rows(15)
    print(df_farm_Cp)


if __name__ == "__main__":
    main()
