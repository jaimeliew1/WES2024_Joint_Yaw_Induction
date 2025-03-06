import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True
FIGPATH = Path(__file__).parent / "debug"

def compare(filename, filename2=None): 
    OLD_PATH = Path(__file__).parent / "old" / filename
    filename2 = filename2 or filename
    NEW_PATH = Path(__file__).parent / "LES_input" / filename2


    with open(OLD_PATH) as f: 
        old = json.load(f)

    with open(NEW_PATH) as f: 
        new = json.load(f)

    ctp_old = [t['ctp'] for t in old['turbines']]
    ctp_new = [t['ctp'] for t in new['turbines']]
    yaw_old = [t['yaw'] for t in old['turbines']]
    yaw_new = [t['yaw'] for t in new['turbines']]

    fig, axs = plt.subplots(ncols=2, figsize=(7, 3))
    ax = axs[0]

    ax.scatter(ctp_old, ctp_new)
    ctpmin = np.min([ctp_old, ctp_new]) - 0.1
    ctpmax = np.max([ctp_old, ctp_new]) + 0.1
    ax.plot([ctpmin, ctpmax], [ctpmin, ctpmax], color='k', lw=0.5)
    ax.set_xlabel("Old $C_T'$ set point")
    ax.set_ylabel("Re-created $C_T'$ setpoint")
    ax.set_title(OLD_PATH.name)

    ax = axs[1]
    ax.scatter(yaw_old, yaw_new)
    yawmin = np.min([yaw_old, yaw_new]) - 0.1
    yawmax = np.max([yaw_old, yaw_new]) + 0.1
    ax.plot([yawmin, yawmax], [yawmin, yawmax], color='k', lw=0.5)
    ax.set_xlabel("Old $\\gamma$ set point (deg.)")
    ax.set_ylabel("Re-created $\\gamma$ setpoint (deg.)")
    # ax.set_title(OLD_PATH.name)

    # plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    plt.savefig(FIGPATH / f"compare_{OLD_PATH.stem}.png", dpi=200)
    plt.close()


if __name__ == "__main__": 
    filename = "diamond_wdir-2.5_{:s}control.json"

    for c in ["yaw", "thrust", "joint"]: 
        compare(filename.format(c))
        # break
