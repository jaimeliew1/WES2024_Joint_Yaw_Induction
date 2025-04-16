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
    "controller_colors",
    "controller_labels",
]

controller_colors = {
    "NoControl": "k",
    "ThrustControl": "tab:blue",
    "YawControl": "tab:orange",
    "YawKOmegaControl": "tab:pink",
    "JointControl": "tab:green",
}


controller_labels = {
    "NoControl": "No Control",
    "ThrustControl": "Thrust Control",
    "YawControl": "Yaw Control",
    "YawKOmegaControl": r"Yaw Control ($K\Omega$)",
    "JointControl": "Joint Control",
}

line_params = {
    "NoControl": dict(
        label=controller_labels["NoControl"], c=controller_colors["NoControl"], ls="--"
    ),
    "ThrustControl": dict(
        label=controller_labels["ThrustControl"], c=controller_colors["ThrustControl"]
    ),
    "YawControl": dict(label=controller_labels["YawControl"], c=controller_colors["YawControl"]),
    # "YawKOmegaControl": dict(
    #     label=controller_labels["YawKOmegaControl"], c=controller_colors["YawKOmegaControl"]
    # ),
    "JointControl": dict(
        label=controller_labels["JointControl"], c=controller_colors["JointControl"]
    ),
}

# fmt: off
DIAMOND_GROUPS = pl.DataFrame(
    {
        "turbine": [0, 4, 24, 20, 1, 9, 23, 15, 2, 14, 22, 10, 5, 3, 19, 21, 6, 8, 18, 16, 7, 13, 17, 11, 12, 12, 12, 12],
        "group": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C", "Dinv", "Dinv", "Dinv", "Dinv", "D", "D", "D", "D", "E", "E", "E", "E", "F", "F", "F", "F"],
        "face": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
    }
)
# fmt: on

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)

FIGDIRFORPAPER = FIGDIR / "for_paper"
FIGDIRFORPAPER.mkdir(exist_ok=True, parents=True)

CACHEDIR = Path(__file__).parent.parent / "data"
# CACHEDIR = Path(r"G:\Shared drives\howland_lab\current_projects\2024_WES_jliew_kheck\data")
CACHEDIR.mkdir(exist_ok=True, parents=True)

ROW_INDICES = [
    [24],
    [23, 19],
    [22, 18, 14],
    [21, 17, 13, 9],
    [20, 16, 12, 8, 4],
    [15, 11, 7, 3],
    [10, 6, 2],
    [5, 1],
    [0],
]

ROW_MAPPING = {
    **dict.fromkeys([0, 5, 10, 15, 20, 21, 22, 23, 24], 1),
    **dict.fromkeys([1, 6, 11, 16, 17, 18, 19], 2),
    **dict.fromkeys([2, 7, 12, 13, 14], 3),
    **dict.fromkeys([3, 9], 4),
    **dict.fromkeys([4], 5),
}


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

    df_by_turbine = df_by_turbine.with_columns(pl.col("wdir").cast(pl.Float64).round(2))

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
    vmin=None,
    vmax=None,
    **kwargs,
):
    angle_rad, r = np.array(angle_rad), np.array(r)
    # Normalise data
    if vmin is None:
        vmin = min(r)
    if vmax is None:
        vmax = max(r)

    midpoint = (vmax + vmin) / 2
    data_range = (vmax - vmin) / 2
    r = (r - midpoint) / data_range * width

    # plot mini wind rose with custom location, radius, and width.
    rose_x = (r + 10) * r0 * (-np.cos(angle_rad)) + x
    rose_y = (r + 10) * r0 * (np.sin(angle_rad)) + y
    ax.plot(rose_x, rose_y, **kwargs)
