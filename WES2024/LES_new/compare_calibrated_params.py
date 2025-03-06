"""
In contrast to `compare_superposition`, where we evaluate
the minimum error in the model form by calibrating directly
to the power data, here we calibrate to the dedicated 
calibration simulation and use the output parameters to 
estimate the power output by the controlled wind farm. 

Kirby Heck
2024 December 16
"""

from pathlib import Path
import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from WES2024.LES_new.compare_superposition import (
    CALIBRATION_RESULTS,
    BASE_LAYOUT,
    CALIBRATION_LAYOUT,
    ROW_INDICES,
    ROW_INDICES_CALIBRATION,
    read_LES_outputs,
    LES_output_dir,
    wakemodels,  # need to import wake models from the other file; names must match
)

figpath = Path(__file__).parent / "calibration"


def run(regenerate=False, calibration_method="nocontrol"):
    """Run script"""
    # load saved data
    df = read_LES_outputs(LES_output_dir.glob("*.csv"))
    df_cached = pl.read_json(CALIBRATION_RESULTS / "apriori_err.json")
    fout = CALIBRATION_RESULTS / f"calibration_params_evaluation_method_{calibration_method}.csv"

    print(df_cached)

    # check if output file already exists
    if fout.exists() and not regenerate:
        print("File", fout, "found, to regenerate, use regenrate=True")
        return

    # Loop through cached calibration set points and perform forward model evals 
    ret = []
    for method, _df in df.group_by("method"):
        print("Evaluating control case: ", method)
        for key, wakemodel in wakemodels.items():
            # set up MITWindfarm
            if method == "calibration_18x3":
                calibration = wakemodel(CALIBRATION_LAYOUT, ROW_INDICES_CALIBRATION)
            else:
                calibration = wakemodel(BASE_LAYOUT, ROW_INDICES)

            # assign set points from LES
            control_setpoints = _df.select("Ctprime", "yaw").to_numpy()
            # extract wake model parameters
            params = np.array(
                df_cached.filter(wakemodel=key,method=calibration_method)["params"]
                .to_list()
            ).squeeze()
            # run MITWindfarm
            sol = calibration.run_windfarm(params, control_setpoints)

            # Compare with LES data at a turbine level and aggregate statistics, store error
            LES_cp = _df["Cp"].to_numpy()
            cost = calibration.cost(params, LES_cp, control_setpoints)
            farm_cp = sol.Cp
            turbine_cp = np.array([rotor.Cp for rotor in sol.rotors])
            mae = np.linalg.norm(LES_cp - turbine_cp, ord=1) / len(LES_cp)
            rmse = np.linalg.norm(LES_cp - turbine_cp, ord=2) / np.sqrt(len(LES_cp))
            # append dictionary to list
            ret.append(
                dict(
                    method=method,
                    wakemodel=key,
                    err=cost,
                    Cp_LES=np.mean(LES_cp),
                    Cp_model=farm_cp,
                    mae=mae,
                    rmse=rmse,
                )
            )

    print("Writing results to ", fout)
    pl.DataFrame(ret).write_csv(fout)


def plot(calibration_method="nocontrol"):
    """Plot error metrics"""
    df = pl.read_csv(CALIBRATION_RESULTS / f"calibration_params_evaluation_method_{calibration_method}.csv")

    df = df.with_columns(
        ((pl.col("Cp_model") - pl.col("Cp_LES")) / pl.col("Cp_LES")).alias("err_farmCp")
    ).filter(pl.col("method").str.contains("yawcontrol"))
    df_agg = df.group_by("wakemodel").agg(pl.col("*").exclude("Cp_LES", "Cp_model").mean())
    df_agg = df_agg.melt(
        id_vars=["method", "wakemodel"], variable_name="metric", value_name="err_val"
    )

    fig = plt.figure(figsize=(6,3))
    sns.barplot(df_agg.sort(by=["metric", "wakemodel"]), x="metric", y="err_val", hue="wakemodel")
    # sns.barplot(df_agg.sort(by=["metric", "err_val"]), x="metric", y="err_val", hue="wakemodel")
    plt.xticks(rotation=80)
    plt.legend(bbox_to_anchor=(1., 0.5), loc="center left", frameon=False)
    plt.subplots_adjust(right=0.7)
    plt.ylim([-0.15, 0.15])
    plt.grid(axis='y', lw=0.75, alpha=0.5)
    plt.savefig(figpath / "calibration_mae.png", bbox_inches="tight", dpi=200)
    plt.close()


if __name__ == "__main__":
    run(regenerate=False, calibration_method="calibration_18x3")
    plot(calibration_method="calibration_18x3")
