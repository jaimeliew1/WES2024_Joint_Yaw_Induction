from pathlib import Path

from WES2024.CustomRotors import UnifiedMomentumLUT
import matplotlib.pyplot as plt
import numpy as np


REGENERATE = True
Ctprimes = np.linspace(-1, 4, 200)
yaws = np.linspace(-30, 30, 101)

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    model = UnifiedMomentumLUT(regenerate=REGENERATE)

    Ctp_grid, yaw_grid = np.meshgrid(Ctprimes, yaws)

    sol = model(Ctp_grid.ravel(), np.deg2rad(yaw_grid.ravel()))

    to_plot = [
        "Cp",
        "Ct",
        "an",
        # "v4",
        # "u4",
        # "dp",
        # "x0",
    ]
    fig, axes = plt.subplots(1, len(to_plot), sharex=True, sharey=True)

    for key, ax in zip(to_plot, axes):
        val = getattr(sol, key).reshape(Ctp_grid.shape)
        ax.imshow(
            val.T,
            origin="lower",
            aspect="auto",
            extent=[
                yaws.min(),
                yaws.max(),
                Ctprimes.min(),
                Ctprimes.max(),
            ],
        )
        ax.set_title(key)

    plt.savefig(FIGDIR / "test_LUT.png", dpi=400, bbox_inches="tight")
