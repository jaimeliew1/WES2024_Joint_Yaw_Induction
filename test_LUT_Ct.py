from pathlib import Path

from WES2024.CustomRotors import ThrustBasedUnifiedMomentumLUT
import matplotlib.pyplot as plt
import numpy as np


REGENERATE = True
Ct = np.linspace(-1, 1.5, 200)
yaws = np.linspace(-50, 50, 101)

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)

to_plot = [
    "Cp",
    "Ctprime",
    "an",
    "v4",
    "u4",
    "dp",
    "x0",
]


def plot_underlying_table():

    df = model.df
    fig, axes = plt.subplots(1, len(to_plot), sharex=True, sharey=True)

    for key, ax in zip(to_plot, axes):
        df_piv = (
            df.sort("Ct", "yaw")
            .pivot(index="yaw", columns="Ct", values=key, maintain_order=True)
            .sort("yaw")
        )
        val = df_piv.to_numpy()[:, 1:].copy()
        ax.imshow(
            val.T,
            origin="lower",
            aspect="auto",
            extent=[
                df["yaw"].min(),
                df["yaw"].max(),
                df["Ct"].min(),
                df["Ct"].max(),
            ],
        )
        ax.set_title(key)

    plt.savefig(FIGDIR / "test_LUT_Ct_underlying.png", dpi=400, bbox_inches="tight")


def plot_interpolated_table():

    Ct_grid, yaw_grid = np.meshgrid(Ct, yaws)

    sol = model(Ct_grid.ravel(), np.deg2rad(yaw_grid.ravel()))

    fig, axes = plt.subplots(1, len(to_plot), sharex=True, sharey=True)

    for key, ax in zip(to_plot, axes):
        val = getattr(sol, key).reshape(Ct_grid.shape)
        ax.imshow(
            val.T,
            origin="lower",
            aspect="auto",
            extent=[
                yaws.min(),
                yaws.max(),
                Ct.min(),
                Ct.max(),
            ],
        )
        ax.set_title(key)

    plt.savefig(FIGDIR / "test_LUT_interpolated.png", dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    model = ThrustBasedUnifiedMomentumLUT(regenerate=REGENERATE)

    plot_underlying_table()
    plot_interpolated_table()
