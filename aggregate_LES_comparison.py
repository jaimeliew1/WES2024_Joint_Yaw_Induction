import json
from pathlib import Path

import polars as pl
from rich import print

data_dir = Path("LES_DATA")
summary_fn = data_dir / "summary.csv"
json_with_layout = data_dir / "MITWindfarm_wdir-2.5_AutoCal_nocontrol.json"


def read_layout(json_fn: Path) -> pl.DataFrame:
    with open(json_fn, "rb") as f:
        asdf = json.load(f)
    dict_list = []
    for i, turbine in enumerate(asdf["turbines"]):
        dict_list.append(dict(turbine_id=i, x=turbine["x"], y=turbine["y"]))
    df_layout = pl.from_dicts(dict_list)
    return df_layout


if __name__ == "__main__":
    df_summary = pl.read_csv(summary_fn)
    df_layout = read_layout(json_with_layout)

    print(df_summary)

    # iterate over individual simulation results and concatenate them to a single dataframe.
    df_list = []
    for simulator, calibration, controller, results_file in df_summary.iter_rows():
        df = (
            pl.read_csv(data_dir / results_file)
            .select(["turbine_id", "Cp", "Ctprime", "yaw"])
            .select(
                pl.lit(simulator).alias("simulator"),
                pl.lit(calibration).alias("calibration"),
                pl.lit(controller).alias("controller"),
                pl.all(),
            )
        )
        df_list.append(df)

    df = pl.concat(df_list)

    # Add a column for the turbine position
    df = df.join(df_layout, on="turbine_id")

    # Write final dataset to file.
    df.write_csv("LES_DATA.csv")
    print(df)
