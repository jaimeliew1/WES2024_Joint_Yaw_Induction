import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np
import polars as pl
plt.rcParams['text.usetex'] = True

BASE = Path(__file__).parent
FIGPATH = BASE / "debug"

def plot(control): 
    OLD_PATH = BASE / "debug" / "wdir-2.5_cosine_{:s}control.json".format(control)
    NEW_PATH = BASE / "debug" / "wdir-2.5_fixedkw_{:s}control.json".format(control)

    with open(OLD_PATH) as f: 
        old = json.load(f)

    with open(NEW_PATH) as f: 
        new = json.load(f)

    ctp_old = [t['ctp'] for t in old['turbines']]
    ctp_new = [t['ctp'] for t in new['turbines']]
    yaw_old = [t['yaw'] for t in old['turbines']]
    yaw_new = [t['yaw'] for t in new['turbines']]

    # plot stuff
    fig, axs = plt.subplots(ncols=2, figsize=(7, 3))
    ax = axs[0]

    ax.scatter(ctp_old, ctp_new)
    ctpmin = np.min([ctp_old, ctp_new]) - 0.1
    ctpmax = np.max([ctp_old, ctp_new]) + 0.1
    ax.plot([ctpmin, ctpmax], [ctpmin, ctpmax], color='k', lw=0.5)
    ax.set_xlabel("JFM $C_T'$ set point")
    ax.set_ylabel("$\\cos^3$ $C_T'$ setpoint")
    ax.set_title(OLD_PATH.stem)

    ax = axs[1]
    ax.scatter(yaw_old, yaw_new)
    yawmin = np.min([yaw_old, yaw_new]) - 0.1
    yawmax = np.max([yaw_old, yaw_new]) + 0.1
    ax.plot([yawmin, yawmax], [yawmin, yawmax], color='k', lw=0.5)
    ax.set_xlabel("JFM $\\gamma$ set point (deg.)")
    ax.set_ylabel("$\\cos^3$ $\\gamma$ setpoint (deg.)")
    ax.set_title(NEW_PATH.stem)

    # plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    plt.savefig(FIGPATH / f"compare_{OLD_PATH.stem}.png", dpi=200)
    plt.close()

def plot_with_polars(control, rotors=["cosine", "jfm", "unified"]): 
    """Plot various set points against the Cosine model set points"""

    for include_TI in [True, False]: 
        rotor = "cosineTI" if include_TI else "cosine"
        base = BASE / "LES_newinput" / f"wdir-2.5_{rotor}_{control}control.csv"
        df_baseline = pl.read_csv(base)

        markers = dict(cosine='x', jfm='v', unified='s')
        
        fig, axs = plt.subplots(ncols=2, figsize=(7,3))
        for rotor in rotors: 
            _rotor = rotor + "TI" if include_TI else rotor
            path = base.parent / f"wdir-2.5_{_rotor}_{control}control.csv"

            df = pl.read_csv(path)
            axs[0].scatter(df_baseline['Ctprime'], df['Ctprime'], label=_rotor, marker=markers[rotor], s=4)
            axs[1].scatter(np.rad2deg(df_baseline['yaw']), np.rad2deg(df['yaw']), label=rotor, marker=markers[rotor], s=4)
        
        axs[0].set_xlabel("$C_T'$ from Cosine model")
        axs[0].set_ylabel("New $C_T'$ set point")
        axs[1].set_xlabel("$\\gamma$ from Cosine model")
        axs[1].set_ylabel("New $\\gamma$ set point")
        axs[0].legend()
        
        plt.tight_layout()
        if include_TI: 
            plt.savefig(FIGPATH / f"compare_cos_jfm_uni_{control}-TI.png", dpi=200)
        else: 
            plt.savefig(FIGPATH / f"compare_cos_jfm_uni_{control}.png", dpi=200)
        plt.close()

if __name__ == "__main__": 
    for control in ["no", "yaw", "thrust", "joint"]: 
        plot_with_polars(control)
        # break
        