"""
Compare unified set points old vs new calibration

Kirby Heck
2024 September 10
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

plt.rcParams["text.usetex"] = True
base = Path(__file__).parent
FIGPATH = base / "debug"


def plot(path1, path2):
    fig, axs = plt.subplots(ncols=2, figsize=(7, 3))

    df1 = read_data(path1)
    df2 = read_data(path2)

    axs[0].scatter(
        df1["Ctprime"],
        df2["Ctprime"],
    )
    axs[1].scatter(
        np.rad2deg(df1["yaw"]),
        np.rad2deg(df2["yaw"]),
    )

    print("MAE Ctprime: ", (df1["Ctprime"] - df2["Ctprime"]).abs().mean())
    print("MAE yaw: ", np.rad2deg(df1["yaw"] - df2["yaw"]).abs().mean())

    axs[0].set_xlabel("$C_T'$ from dataset 1")
    axs[0].set_ylabel("$C_T'$ from dataset 2")
    axs[1].set_xlabel("$\\gamma$ from dataset 1")
    axs[1].set_ylabel("$\\gamma$ from dataset 2")

    for ax in axs:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmin = np.min([xlim, ylim])
        xmax = np.max([xlim, ylim])
        ax.plot([xmin, xmax], [xmin, xmax], color="k", lw=0.5)

    axs[0].set_title(f"dataset 1: {path1.stem}", fontsize=9)
    axs[1].set_title(f"dataset 2: {path2.stem}", fontsize=9)

    plt.tight_layout()
    return axs


def scatter_setpoints(path, ax=None, use_abs_yaw=True, **ls):
    if ax is None:
        fig, ax = plt.subplots()

    data = read_data(path)
    yaws = abs(np.rad2deg(data["yaw"])) if use_abs_yaw else np.rad2deg(data["yaw"])
    ax.scatter(yaws, data["Ctprime"], **ls)
    return ax


def read_data(fpath):
    fpath = Path(fpath)
    if fpath.suffix == ".json":
        with open(fpath, "r") as f:
            src = json.load(f)
        df = pl.DataFrame(src["turbines"])
        return df.with_columns(df["ctp"].alias("Ctprime"), np.deg2rad(df["yaw"]))
    elif fpath.suffix == ".csv":
        return pl.read_csv(fpath)
    else:
        raise NotImplementedError


def plot_uni_oldnew():
    plot(
        base / "LES_archive" / "diamond_wdir-2.5_jointunicontrol.json",
        base / "LES_newinput" / "diamond_wdir-2.5_unifiedTI_jointcontrol.csv",
    )
    plt.savefig(FIGPATH / "compare_jointuni_old_new.png", dpi=300)
    plt.close()


def plot_jfm_oldnew():
    plot(
        base / "LES_archive" / "diamond_wdir-2.5_jointcontrol.json",
        base / "LES_newinput" / "diamond_wdir-2.5_jfmTI_jointcontrol.csv",
    )
    plt.savefig(FIGPATH / "compare_jointjfm_old_new.png", dpi=300)
    plt.close()


def plot_scatter():
    """I think this function is broken KSH 2024-12-10"""
    markers = ['o', 'v', 's']
    colors=['tab:blue', 'tab:red', 'k']

    fig, ax = plt.subplots(figsize=(4,3))
    for k, _path in enumerate((base / "LES_newinput").glob("*unified*joint*.csv")):
        rotormodel = (_path.name).split("diamond_wdir-2.5_")[1].split("TI")[0]
        scatter_setpoints(_path, ax=ax, label=rotormodel, marker=markers[k], c=colors[k], s=5)
    
    ax.legend()
    ax.set_xlabel("Yaw set point $|\\gamma|$")
    ax.set_ylabel("Thrust $C_T'$")
    plt.tight_layout()
    # plt.savefig(FIGPATH / "rotormodel_setpoint_scatter.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_scatter()
    # plot_jfm_oldnew()
    # plot_uni_oldnew()
