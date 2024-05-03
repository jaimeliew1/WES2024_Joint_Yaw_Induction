from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import polars as pl

from mitwindfarm.Layout import Layout
from rich import print

from WES2024.LES_calibration_backend import (
    generate_individual_cal,
    generate_model_cal,
    run_model,
    normalize_by_upstream,
    LAYOUT,
)
from WES2024 import utils

REGENERATE = True

FILESTEM = Path(__file__).stem
LES_FN = Path("data/mean_power_wdir-2.5_5hr.csv")

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)


@utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False) -> pl.DataFrame:
    df_LES = pl.read_csv(LES_FN).with_columns(method=pl.lit("LES"), TI=np.nan, kw=np.nan)
    df_LES = df_LES.select(
        "turbine_id",
        "Cp",
        pl.Series(normalize_by_upstream(df_LES["Cp"].to_numpy())).alias("P_norm"),
        "method",
        "TI",
        "kw",
    )

    sol = generate_individual_cal()
    df_indiv = pl.DataFrame(
        dict(
            turbine_id=np.arange(0, 25),
            Cp=[x.Cp for x in sol.rotors],
            P_norm=normalize_by_upstream(np.array([x.Cp for x in sol.rotors])),
            method="individual_cal",
            TI=[x.TI for x in sol.rotors],
            kw=[x.kw for x in sol.wakes],
        )
    )

    sol = generate_model_cal()
    df_model = pl.DataFrame(
        dict(
            turbine_id=np.arange(0, 25),
            Cp=[x.Cp for x in sol.rotors],
            P_norm=normalize_by_upstream(np.array([x.Cp for x in sol.rotors])),
            method="model_cal",
            TI=[x.TI for x in sol.rotors],
            kw=[x.kw for x in sol.wakes],
        )
    )

    sol = generate_model_cal(upstream_normalisation=True)
    df_model_norm = pl.DataFrame(
        dict(
            turbine_id=np.arange(0, 25),
            Cp=[x.Cp for x in sol.rotors],
            P_norm=normalize_by_upstream(np.array([x.Cp for x in sol.rotors])),
            method="model_cal_upstream_norm",
            TI=[x.TI for x in sol.rotors],
            kw=[x.kw for x in sol.wakes],
        )
    )

    sol = run_model(0, 0, 0.07)
    df_uncal = pl.DataFrame(
        dict(
            turbine_id=np.arange(0, 25),
            Cp=[x.Cp for x in sol.rotors],
            P_norm=normalize_by_upstream(np.array([x.Cp for x in sol.rotors])),
            method="uncalibrated",
            TI=[x.TI for x in sol.rotors],
            kw=[x.kw for x in sol.wakes],
        )
    )
    df = pl.concat([df_LES, df_indiv, df_model, df_model_norm, df_uncal])
    return df
    # breakpoint()


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
    plt.close()


def plot(df: pl.DataFrame):
    for method, _df in df.group_by("method"):
        plot_text_on_layout(
            LAYOUT,
            _df["kw"].to_numpy(),
            utils.FIGDIR / f"{FILESTEM}_kw_{method}.png",
            f"kw ({method})",
        )
        plot_text_on_layout(
            LAYOUT,
            _df["TI"].to_numpy(),
            utils.FIGDIR / f"{FILESTEM}_TI_{method}.png",
            f"TI ({method})",
        )

        diff = (_df["Cp"].sum() - df.filter(method="LES")["Cp"].sum()) / df.filter(method="LES")[
            "Cp"
        ].sum()
        diff_norm = (_df["P_norm"].sum() - df.filter(method="LES")["P_norm"].sum()) / df.filter(
            method="LES"
        )["P_norm"].sum()
        plot_text_on_layout(
            LAYOUT,
            _df["P_norm"].to_numpy(),
            utils.FIGDIR / f"{FILESTEM}_P_norm_{method}.png",
            rf"Cp ({method}) (farm C_P error: {diff*100:2.2f}\%), (normalized C_P error:  {diff_norm*100:2.2f}\%)",
        )

    plt.figure()
    for method, _df in df.group_by("method"):
        plt.plot(_df["TI"], _df["kw"], ".", label=f"{method}")
    plt.xlabel("TI")
    plt.ylabel("$k_w$")
    plt.title(f"{method}")
    plt.legend()
    plt.savefig(utils.FIGDIR / f"{FILESTEM}_kw_vs_TI.png", dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    df = generate(regenerate=REGENERATE)
    print(df)

    plot(df)
