"""
Create two-turbine power contours

Kirby Heck
2025 March 07
"""

import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from pathlib import Path
import WES2024  # adds latex font

ROOTDIR = Path(__file__).parent
DATA_LES = ROOTDIR / "twoturbine_alldata_LES.csv"
DATA_UNIFIED = ROOTDIR / "twoturbine_mitwindfarm_unified.csv"
DATA_COSINE = ROOTDIR / "twoturbine_mitwindfarm_cosine.csv"
DATA_COSINE3 = ROOTDIR / "twoturbine_mitwindfarm_cosine3.csv"

FIGDIR = ROOTDIR / "fig"
FIGDIR.mkdir(parents=True, exist_ok=True)


def plot_contour(ax, yaws, ctps, Cps, **plt_kwargs): 
    """Plots contours, returns image object."""
    df = pl.DataFrame(dict(yaw=yaws, ctp=ctps, Cp=Cps)).sort(by=['ctp', 'yaw'])
    nyaw = len(df['yaw'].unique())
    nctp = len(df['ctp'].unique())
    yawG = df['yaw'].to_numpy().reshape((nctp, nyaw))
    ctpG = df['ctp'].to_numpy().reshape((nctp, nyaw))
    CpG = df['Cp'].to_numpy().reshape((nctp, nyaw))

    default_kwargs = dict(vmin=0.3, vmax=0.6, extend="both", levels=np.arange(0.3, 0.61, 0.025), cmap='cividis')
    default_kwargs.update(plt_kwargs)

    return ax.contourf(yawG, ctpG, CpG, **default_kwargs)  # returns image object


def plot_max(ax, yaws, ctps, Cps, **plt_kwargs): 
    """Plots a marker at the maximum Cp value"""
    _id = np.argmax(Cps)
    return ax.scatter(yaws.to_numpy()[_id], ctps.to_numpy()[_id], **plt_kwargs)

def make_plots(): 
    unified = pl.read_csv(DATA_UNIFIED)
    cosine = pl.read_csv(DATA_COSINE3)
    les = pl.read_csv(DATA_LES)

    plot_order = ['Cp_T', 'Cp_1', 'Cp_2']
    plot_label = dict(Cp_T="$C_{P,\\mathrm{farm}}$", Cp_1="$C_{P,1}$", Cp_2="$C_{P,2}$")
    plot_titles = dict(LES=les, Unified=unified, Cosine=cosine)

    fig, axarr = plt.subplots(figsize=(2*len(plot_titles)+0.5, 5.5), nrows=3, ncols=len(plot_titles), sharex=True, sharey=True)
    k = 0
    for keyval, axs in zip(plot_titles.items(), axarr.T): 
        src, df = keyval
        for key, ax in zip(plot_order, axs): 
            # label subfigures
            ax.text(0, 1.02, f"{chr(k+97)})", ha='center', va='bottom', transform=ax.transAxes, fontsize=10)
            k += 1

            # plot contours
            im = plot_contour(ax, df['yaw'], df['ctp'], df[key], cmap='viridis', )
            if src == "LES": 
                ax.scatter(df['yaw'], df['ctp'], s=5, marker='o', color='w', edgecolor='k', lw=0.5)
                for ax in axarr.ravel(): 
                    plot_max(ax, df['yaw'], df['ctp'], df['Cp_T'], color='tab:blue', marker='*', lw=0.5, edgecolor='w', zorder=10, s=50)
            else: 
                plot_max(ax, df['yaw'], df['ctp'], df['Cp_T'], color='tab:red', marker='s', lw=0.5, edgecolor='w', s=25)


    for ax in axarr[:, 0]: 
        ax.set_ylabel("$C_{T,1}'$")
    for ax in axarr[-1, :]: 
        ax.set_xlabel("$\\gamma_1$ (deg.)")
    for axs, key in zip(axarr, plot_order):  # label colorbars
        fig.colorbar(im, ax=axs, label=plot_label[key])
    for ax, title in zip(axarr[0, :], plot_titles.keys()):
        ax.set_title(title)
        
    plt.savefig(FIGDIR / "twoturbine_contours.png", dpi=300)
    plt.savefig(FIGDIR / "twoturbine_contours.pdf")
    plt.show()


def make_plot_1row(): 
    unified = pl.read_csv(DATA_UNIFIED)
    cosine = pl.read_csv(DATA_COSINE3)
    les = pl.read_csv(DATA_LES)

    key = 'Cp_T'
    plot_label = dict(Cp_T="Farm efficiency", Cp_1="$C_{P,1}$", Cp_2="$C_{P,2}$")
    plot_titles = dict(CFD=les, Classical=cosine, Unified=unified)

    fig, axs = plt.subplots(figsize=(7, 2.25), nrows=1, ncols=len(plot_titles), sharex=True, sharey=True)
    k = 0
    for keyval, ax in zip(plot_titles.items(), axs): 
        src, df = keyval
        # # label subfigures
        # ax.text(0, 1.02, f"{chr(k+97)})", ha='center', va='bottom', transform=ax.transAxes, fontsize=10)
        # k += 1

        # plot contours
        im = plot_contour(ax, df['yaw'], df['ctp'], df[key], cmap='viridis', )
        if src == "CFD": 
            ax.scatter(df['yaw'], df['ctp'], s=5, marker='o', color='w', edgecolor='k', lw=0.5)
            for ax in axs: 
                plot_max(ax, df['yaw'], df['ctp'], df['Cp_T'], color='tab:blue', marker='*', lw=0.5, edgecolor='w', zorder=10, s=50)
        else: 
            plot_max(ax, df['yaw'], df['ctp'], df['Cp_T'], color='tab:red', marker='s', lw=0.5, edgecolor='w', s=25)


    plt.subplots_adjust(bottom=0.4, right=0.8)
    axs[0].set_ylabel("Thrust $C_T'$")
    for ax in axs: 
        ax.set_xlabel("Yaw (degrees)")
    fig.colorbar(im, ax=axs, label=plot_label[key])
    for ax, title in zip(axs, plot_titles.keys()):
        ax.set_title(title)
    
    axs[1].scatter([], [], color='tab:red', marker='s', lw=0.5, edgecolor='w', s=25, label="Model optimum")
    axs[1].scatter([], [], color='tab:blue', marker='*', lw=0.5, edgecolor='w', zorder=10, s=50, label="CFD optimum")
    axs[1].scatter([], [], s=5, marker='o', color='w', edgecolor='k', lw=0.5, label="CFD simulation")
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncols=3, fontsize=10)
    
    plt.savefig(FIGDIR / "twoturbine_contours_1row.png", dpi=300)
    plt.savefig(FIGDIR / "twoturbine_contours_1row.pdf")
    plt.show()


if __name__ == "__main__":
    """Make plots"""
    make_plots()