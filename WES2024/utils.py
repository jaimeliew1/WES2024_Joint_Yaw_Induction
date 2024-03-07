import functools
import pickle
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mitwindfarm.windfarm import Windfarm, WindfarmSolution
from numpy.typing import ArrayLike

__all__ = [
    "FIGDIR",
    "CACHEDIR",
    "to_polars",
    "from_polars",
    "cache_pickle",
    "cache_polars",
]
# fmt: off
DIAMOND_GROUPS = pl.DataFrame(
    {
        "turbine": [0, 4, 24, 20, 1, 9, 23, 15, 2, 14, 22, 10, 5, 3, 19, 21, 6, 8, 18, 16, 7, 13, 17, 11, 12, 12, 12, 12],
        "group": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C", "D", "D", "D", "D", "E", "E", "E", "E", "F", "F", "F", "F", "G", "G", "G", "G"],
        "face": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
    }
)
# fmt: on

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)

CACHEDIR = Path(__file__).parent.parent / "data"
CACHEDIR.mkdir(exist_ok=True, parents=True)


def fill_in_other_quadrants(
    df: pl.DataFrame, diamond_groups: pl.DataFrame = DIAMOND_GROUPS
) -> pl.DataFrame:
    # First generate full wind rose for each turbine group.
    df_by_group = []
    for turbine, group, face in diamond_groups.iter_rows():
        _df = (
            df.filter(pl.col("turbine") == turbine)
            .with_columns(pl.col("wdir") + face * 90, pl.lit(group).alias("group"))
            .select(pl.exclude("turbine", "x", "y", "z"))
        )
        df_by_group.append(_df)

    df_by_group = pl.concat(df_by_group)

    # Next, fill out the wind rose for each turbine.
    df_by_turbine = []
    seen_turbines = []
    for turbine, group, face in diamond_groups.iter_rows():
        if turbine in seen_turbines:
            continue
        seen_turbines.append(turbine)

        _df = df_by_group.filter(pl.col("group") == group).with_columns(
            (pl.col("wdir") + 360.0 - face * 90).mod(360.0), pl.lit(turbine).alias("turbine")
        )
        df_by_turbine.append(_df)
    df_by_turbine = pl.concat(df_by_turbine)

    return df_by_turbine


def to_polars(sol: WindfarmSolution) -> pl.DataFrame:
    """
    Convert a WindfarmSolution object to a Polars DataFrame.

    Parameters:
    - sol (WindfarmSolution): The WindfarmSolution object to be converted.

    Returns:
    - pl.DataFrame: The Polars DataFrame containing turbine layout, setpoints, and rotor solutions.
    """
    out = []
    for i, ((x, y, z), setpoint, rotor_sol) in enumerate(
        zip(sol.layout, sol.setpoints, sol.rotors)
    ):
        _out = (
            dict(turbine=i, x=x, y=y, z=z)
            | asdict(rotor_sol)
            | {f"setpoint_{j}": s for j, s in enumerate(setpoint)}
        )

        out.append(_out)

    return pl.from_dicts(out).drop("extra")


def from_polars(df: pl.DataFrame, windfarm_model: Windfarm) -> WindfarmSolution:
    """
    Convert a Polars DataFrame to a WindfarmSolution object.

    Parameters:
    - df (pl.DataFrame): The Polars DataFrame containing turbine layout, setpoints, and rotor solutions.
    - windfarm_model (Windfarm): The Windfarm model to use for creating the WindfarmSolution.

    Returns:
    - WindfarmSolution: The reconstructed WindfarmSolution object.
    """
    assert len(df) == len(df["turbine"].unique())
    df = df.sort("turbine")
    partial = dict(layout=[], setpoints=[], rotors=[])
    for _df in df.iter_rows(named=True):
        partial["layout"].append((_df["x"], _df["y"], _df["z"]))
        partial["setpoints"].append(tuple(v for k, v in _df.items() if k.startswith("setpoint_")))
        partial["rotors"].append(
            {key: _df[key] for key in ["yaw", "Cp", "Ct", "Ctprime", "an", "u4", "v4", "REWS"]}
        )

    return windfarm_model.from_dict(partial)


def cache_pickle(cache_file: str | Path):
    """
    Decorator function for caching data using pickle.

    Parameters:
    - cache_file (Union[str, Path]): The path to the cache file.

    Returns:
    - Callable: Decorator function to be applied to another function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_filepath = Path(cache_file)
            cache_filepath.parent.mkdir(exist_ok=True, parents=True)
            regenerate = kwargs.pop("regenerate", False)

            # Check if the cache file exists and regeneration is not forced
            if not regenerate and cache_filepath.exists():
                print(f"Loading data from cache: {cache_filepath}")
                with open(cache_filepath, "rb") as file:
                    return pickle.load(file)
            else:
                # Generate and save the data
                data = func(*args, **kwargs)
                print(f"Saving data to cache: {cache_filepath}")
                with open(cache_filepath, "wb") as file:
                    pickle.dump(data, file)
                return data

        return wrapper

    return decorator


def cache_polars(cache_file: str | Path):
    """
    Decorator function for caching Polars DataFrame using CSV format.

    Parameters:
    - cache_file (Union[str, Path]): The path to the cache file.

    Returns:
    - Callable: Decorator function to be applied to another function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_filepath = Path(cache_file)
            cache_filepath.parent.mkdir(exist_ok=True, parents=True)
            regenerate = kwargs.pop("regenerate", False)

            # Check if the cache file exists and regeneration is not forced
            if not regenerate and cache_filepath.exists():
                print(f"Loading data from cache: {cache_filepath}")
                return pl.read_csv(cache_filepath)
            else:
                # Generate and save the data
                df = func(*args, **kwargs)
                print(f"Saving data to cache: {cache_filepath}")
                df.write_csv(cache_filepath)
                return df

        return wrapper

    return decorator


def my_polar_plot(
    angle_rad: ArrayLike,
    r: ArrayLike,
    x: float,
    y: float,
    r0: float,
    width: float,
    ax: plt.Axes,
    lw: float = 0.5,
    style: str = None,
):
    angle_rad, r = np.array(angle_rad), np.array(r)
    # Normalise data
    r = (r - np.mean(r)) / np.max(np.abs(r)) * width

    # plot mini wind rose with custom location, radius, and width.
    rose_x = (r + 10) * r0 * (-np.cos(angle_rad)) + x
    rose_y = (r + 10) * r0 * (np.sin(angle_rad)) + y
    ax.plot(rose_x, rose_y, style, lw=lw)
