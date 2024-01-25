from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from foreach import foreach
from mitwindfarm import Plotting
from mitwindfarm.windfarm import Windfarm, WindfarmSolution
from utilities import from_polars
import ffmpeg

from plot_02_spiral import generate

import tempfile

# Using TemporaryDirectory as a context manager


FIGDIR = Path(__file__).parent.parent / "fig/"
FIGDIR.mkdir(exist_ok=True, parents=True)

CACHEDIR = FIGDIR / "animation_cache"
CACHEDIR.mkdir(exist_ok=True, parents=True)

windfarm = Windfarm()


def animate(dir_to_animate, out_fn, framerate=10, wildcard="/*.png"):

    dir_to_animate = Path(dir_to_animate)
    N_files = len(list(dir_to_animate.iterdir()))

    print(f"animating {N_files} frames...")

    (
        ffmpeg.input(
            dir_to_animate.as_posix() + wildcard,
            pattern_type="glob",
            framerate=framerate,
        )
        .output(out_fn.as_posix())
        .run(overwrite_output=True, quiet=True)
    )


def _plot_single(x):
    i, _df, wdir = x

    fn = CACHEDIR / f"{i:04}.png"
    windfarm_sol = from_polars(_df, windfarm)
    plot_single(windfarm_sol, fn)


def plot_single(windfarm_sol: WindfarmSolution, fn: Path):
    Plotting.plot_windfarm(windfarm_sol, frame=False)
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def main(method="JointControl", min_dist=4):
    df: pl.DataFrame = generate(regenerate=False)

    df = df.filter(pl.col("method") == method).filter(pl.col("min_dist") == min_dist)

    params = [
        (i, _df, wdir)
        for i, (wdir, _df) in enumerate(
            df.sort("wdir").group_by("wdir", maintain_order=True)
        )
    ]

    foreach(_plot_single, params, parallel=False)

    animate(CACHEDIR, FIGDIR / "animation.mp4", framerate=10)


if __name__ == "__main__":
    main()
