"""
Compare cosine versus most recent model calibration
for Mike's talk 2025 February. 

Cosine data was run in September 2024 and the most recent
model calibration + LES was run in January 2024. 

Kirby Heck
2025 February 11
"""

import polars as pl
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# need to generate set points from consistent wake models...
# makes sense to use the most recent calibration for this


from mitwindfarm import Windfarm, Layout, Niayifar
from WES2024.CustomRotors import UnifiedLUTAD, CosineAD
from WES2024.LES.run import TIAMB
from WES2024.LES_new.compare_superposition import (
    VariableKwGaussianWakeModel,
    CALIBRATION_RESULTS,
)
from WES2024.LES.cosine_setpoints import UniGauss
from plot_row_power import ROW_MAPPING


FIGPATH = Path(__file__).parent / "figs"
LES_OUTPUTDIR = Path(__file__).parent / "LES_output"

LES_TO_PLOT = [
    "LES_wdir-2.5_LESnew_nocontrol.csv",
    "LES_wdir-2.5_LESnew_yawcontrol.csv",
    "LES_wdir-2.5_LESnew_jointcontrol.csv",
    "LESfake_wdir-2.5_cosineTI_nocontrol.csv",  # this is a duplicate "no control" case
    "LES_wdir-2.5_cosineTI_yawcontrol.csv",
    "LES_wdir-2.5_cosineTI_jointcontrol.csv",
]

NAMING_KEYS = {
    "LESnew_nocontrol": "No Control",
    "LESnew_yawcontrol": "Unified + \nYaw ctrl.",
    "cosineTI_nocontrol": "No Control",
    "LESnew_jointcontrol": "Unified + \nJoint ctrl.",
    "cosineTI_yawcontrol": "Cosine + \nYaw ctrl.",
    "cosineTI_jointcontrol": "Cosine + \nJoint ctrl.",
}

ROTORS = {"cosineTI": CosineAD(Pp=3), "LESnew": UnifiedLUTAD()}


def make_windfarm(rotor, modelname="04_kw_TI"):
    """Create a windfarm object with the correct wake model"""
    # load calibrated wake model parameters
    df_cached = pl.read_json(CALIBRATION_RESULTS / "final_calibration_parameters.json")
    params = np.array(
        df_cached.filter(wakemodel=modelname, method="LESnew_nocontrol")["params"].to_list()
    ).squeeze()

    if rotor == "cosineTI":
        # unified setpoints from calibration
        wakemodel = UniGauss
    else:
        wakemodel = VariableKwGaussianWakeModel(*params, x0=3)

    return Windfarm(
        rotor_model=ROTORS[rotor], wake_model=wakemodel, TIamb=TIAMB, superposition=Niayifar()
    )


def compute_windfarm(df_LES, rotor):
    """Solve the windfarm for a windfarm solution"""
    windfarm = make_windfarm(rotor=rotor)
    layout = Layout(df_LES["x"].to_numpy(), df_LES["y"].to_numpy())
    setpoints = [(ctp, yaw) for ctp, yaw in df_LES.select(["Ctprime", "yaw"]).iter_rows()]
    wfsol = windfarm(layout, setpoints)
    return wfsol


def debug():
    """
    Trying to match the old set cosine model power predictions
    with the old optimized set points is driving me crazy.

    What models were used? They should all be the exact same
    as in `cosine_setpoints.py`!!! arg.
    """
    df1 = pl.read_csv(LES_OUTPUTDIR / "LES_wdir-2.5_cosineTI_jointcontrol.csv")
    wfsol = compute_windfarm(df1, rotor="cosineTI")
    Cps = [r.Cp for r in wfsol.rotors]

    df_ref = pl.read_csv(
        Path(__file__).parent.parent / "LES" / "LES_newinput" / "wdir-2.5_cosineTI_jointcontrol.csv"
    )
    df_ref = df_ref.with_columns(
        pl.Series("Cp_current", Cps),
        pl.col("turbine").replace(ROW_MAPPING, default=None).alias("row"),
    )
    fig, ax = plt.subplots()
    ax.scatter(df_ref["Cp"], Cps, c=df_ref["row"], label="Ref")
    ax.plot([0.2, 0.6], [0.2, 0.6], color="k", lw=0.5)
    plt.show()


def plot(figpath=FIGPATH / "LES_Cp_gain_Cosine", exts=[".png"]):  # , ".pdf", ".eps", ".svg"]):
    """Plot power gain in yaw, joint control"""
    results = []
    for les_output in LES_TO_PLOT:
        fname = LES_OUTPUTDIR / les_output
        df_LES = pl.read_csv(fname)
        name = fname.stem.split("wdir-2.5_")[1]
        rotor, controller = name.split("_")
        wfsol = compute_windfarm(df_LES, rotor=rotor)

        # I hate working with this data...
        df_LES = (
            df_LES.mean()
            .with_columns(
                pl.lit("LES").alias("simulator"),
                pl.lit(NAMING_KEYS[name]).alias("name"),
                pl.lit(controller).alias("controller"),
                pl.lit(rotor).alias("rotor"),
            )
            .select("name", "controller", "rotor", "simulator", "Cp")
        )
        df_model = pl.DataFrame(
            {
                "name": NAMING_KEYS[name],
                "controller": controller,
                "rotor": rotor,
                "simulator": "Model",
                "Cp": wfsol.Cp,
            }
        )
        results += [df_LES, df_model]

    # now compute Cp gain
    df = pl.concat(results)
    norm_values = (
        df.filter(pl.col("controller") == "nocontrol")
        .group_by("simulator", "rotor")
        .mean()
        .select(["simulator", "rotor", "Cp"])
        .rename({"Cp": "Cp_norm"})
    )
    df = (
        df.join(norm_values, on=["simulator", "rotor"])
        .with_columns((100 * (pl.col("Cp") / pl.col("Cp_norm") - 1)).alias("Power gain (\\%)"))
        .filter(pl.col("controller") != "nocontrol")
        .rename({"simulator": "Simulator"})
        .sort(by=["Simulator", "rotor", "controller"], descending=True)
    )

    fig, ax = plt.subplots(figsize=(4, 3))
    barplot = sns.barplot(
        df,
        x="name",
        y="Power gain (\\%)",
        hue="Simulator",
        palette=("tab:purple", "tab:green"),
        lw=2,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.axhline(0, color="k", lw=0.75)
    for bar in barplot.patches:
        # Get the current face color
        facecolor = bar.get_facecolor()

        # Set the facecolor with the desired alpha
        bar.set_edgecolor(facecolor)
        bar.set_facecolor((facecolor[0], facecolor[1], facecolor[2], 0.5))

    plt.tight_layout()
    for ext in exts:
        plt.savefig(figpath.with_suffix(ext), dpi=600)
    print("Saving figure to", figpath)


def plot_4stage(fname_base="LES_Cp_gain_cosine", exts=[".png"]):
    """Plot power gain in yaw, joint control"""
    results = []
    NAMING_KEYS = {
        "LESnew_nocontrol": "No Control",
        "LESnew_yawcontrol": "New + \nYaw ctrl.",
        "cosineTI_nocontrol": "No Control",
        "LESnew_jointcontrol": "New + \nJoint ctrl.",
        "cosineTI_yawcontrol": "Empirical + \nYaw ctrl.",
        "cosineTI_jointcontrol": "Empirical + \nJoint ctrl.",
    }

    for les_output in LES_TO_PLOT:
        fname = LES_OUTPUTDIR / les_output
        df_LES = pl.read_csv(fname)
        name = fname.stem.split("wdir-2.5_")[1]
        rotor, controller = name.split("_")
        wfsol = compute_windfarm(df_LES, rotor=rotor)

        # I hate working with this data...
        df_LES = (
            df_LES.mean()
            .with_columns(
                pl.lit("CFD").alias("simulator"),
                pl.lit(NAMING_KEYS[name]).alias("name"),
                pl.lit(controller).alias("controller"),
                pl.lit(rotor).alias("rotor"),
            )
            .select("name", "controller", "rotor", "simulator", "Cp")
        )
        df_model = pl.DataFrame(
            {
                "name": NAMING_KEYS[name],
                "controller": controller,
                "rotor": rotor,
                "simulator": "Model",
                "Cp": wfsol.Cp,
            }
        )
        results += [df_LES, df_model]

    # now compute Cp gain
    df = pl.concat(results)
    norm_values = (
        df.filter(pl.col("controller") == "nocontrol")
        .group_by("simulator", "rotor")
        .mean()
        .select(["simulator", "rotor", "Cp"])
        .rename({"Cp": "Cp_norm"})
    )
    df = (
        df.join(norm_values, on=["simulator", "rotor"])
        .with_columns((100 * (pl.col("Cp") / pl.col("Cp_norm") - 1)).alias("Power gain (\\%)"))
        .filter(pl.col("controller") != "nocontrol")
        .rename({"simulator": "Simulator"})
        .sort(by=["Simulator", "rotor", "controller"], descending=True)
    )

    filters = [
        ((pl.col("rotor") == "cosineTI") & (pl.col("Simulator") == "Model"),),
        ((pl.col("rotor") == "cosineTI"),),
        ((pl.col("rotor") != "LESnew").or_(pl.col("Simulator") != "CFD"),),
        None,
    ]
    for k in range(4):
        fig, ax = plt.subplots(figsize=(4, 3))
        df_sub = df.filter(*filters[k]) if filters[k] else df

        # hotfix to get column widths consistent
        if len(df_sub['Simulator'].unique()) < 2: 
            df_sub = pl.concat(
                [
                    df_sub,
                    pl.DataFrame(
                        {
                            "name": df_sub["name"].unique(),
                            "Power gain (\\%)": np.nan,
                            "Simulator": df["Simulator"].unique(),
                        }
                    ),
                ], 
                how="diagonal_relaxed", 
            )

        barplot = sns.barplot(
            df_sub,
            x="name",
            y="Power gain (\\%)",
            hue="Simulator",
            palette=("tab:purple", "tab:green"),
            lw=2,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_xlim(np.array([0, len(df["name"].unique())]) - 0.5)
        ax.tick_params(axis='x', labelsize=9)

        ax.set_ylim([0, df["Power gain (\\%)"].max() + 1])
        ax.axhline(0, color="k", lw=0.75)
        ax.legend(loc="upper left", title="Simulator")
        for bar in barplot.patches:
            # Get the current face color
            facecolor = bar.get_facecolor()

            # Set the facecolor with the desired alpha
            bar.set_edgecolor(facecolor)
            bar.set_facecolor((facecolor[0], facecolor[1], facecolor[2], 0.5))

        plt.tight_layout()
        figpath = FIGPATH / (fname_base + f"_{k}")
        for ext in exts:
            plt.savefig(figpath.with_suffix(ext), dpi=600)
        print("Saving figure to", figpath)


if __name__ == "__main__":
    plot_4stage()
    print("Done")
