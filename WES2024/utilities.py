from dataclasses import asdict
import polars as pl
from mitwindfarm.windfarm import Windfarm, WindfarmSolution
import numpy as np


def to_polars(sol: WindfarmSolution) -> pl.DataFrame:
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

    return pl.from_dicts(out)


def from_polars(df: pl.DataFrame, windfarm_model: Windfarm) -> WindfarmSolution:
    assert len(df) == len(df["turbine"].unique())
    df = df.sort("turbine")
    partial = dict(layout=[], setpoints=[], rotors=[])
    for _df in df.iter_rows(named=True):
        partial["layout"].append((_df["x"], _df["y"], _df["z"]))
        partial["setpoints"].append(
            tuple(v for k, v in _df.items() if k.startswith("setpoint_"))
        )
        partial["rotors"].append(
            {
                key: _df[key]
                for key in ["yaw", "Cp", "Ct", "Ctprime", "an", "u4", "v4", "REWS"]
            }
        )

    return windfarm_model.from_dict(partial)


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
