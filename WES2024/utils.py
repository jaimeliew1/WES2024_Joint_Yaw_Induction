import functools
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
from mitwindfarm.windfarm import Windfarm, WindfarmSolution


__all__ = [
    "FIGDIR",
    "CACHEDIR",
    "to_polars",
    "from_polars",
    "cache_pickle",
    "cache_polars",
]


FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)

CACHEDIR = Path(__file__).parent.parent / "data"
CACHEDIR.mkdir(exist_ok=True, parents=True)


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


def hankel_transform(X, s):
    """
    stacks the snapshots, X, so that each new snapshot contains the previous
    s snapshots.
    args:
        X (2D array): n by m matrix.
        s (int): stack size
    returns:
        Xout (2D array): n - (k-1) by k*m matrix.
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if s == 1:
        return X
    l, m = X.shape
    w = m - (s - 1)
    out = np.zeros([l * s, w])

    for i in range(s):
        row = X[:, m - i - w : m - i]
        out[i * l : (i + 1) * l, :] = row

    return out


def truncatedSVD(X, r):
    """
    Computes the truncated singular value decomposition (SVD)
    args:
        X (2d array): Matrix to perform SVD on.
        rank (int or float): rank parameter of the svd. If a positive integer,
        truncates to the largest r singular values. If a float such that 0 < r < 1,
        the rank is the number of singular values needed to reach the energy
        specified in r. If -1, no truncation is performed.
    """

    U, S, V = np.linalg.svd(X, full_matrices=False)
    V = V.conj().T
    if r >= 1:
        rank = min(r, U.shape[1])

    elif 0 < r < 1:
        cumulative_energy = np.cumsum(S**2 / np.sum(S**2))
        rank = np.searchsorted(cumulative_energy, r) + 1

    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = V[:, :rank]

    return U_r, S_r, V_r
