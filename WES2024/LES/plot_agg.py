from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl


data_dir = Path(__file__).parent / "LES_output"
fig_dir = Path(__file__).parent / "fig"
fig_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Load data and aggregate into single dataframe
    df_list = []
    for fn in data_dir.glob("*.csv"):
        _df = pl.read_csv(fn)

        if fn.name.startswith("LES"):
            wdir = float(fn.name.split("wdir")[1].split("_")[0])
            controller = fn.name.split("_")[-1].split(".")[0]
            _df = _df.with_columns(
                simulator=pl.lit("LES"), controller=pl.lit(controller), wdir=wdir
            ).rename({"turbine_id": "turbine"})

        df_list.append(_df.select(["simulator", "controller", "wdir", "turbine", "Cp"]))

    df = pl.concat(df_list)
    print(df)

    # Calculate farm power and normalize by nocontrol controller as a reference.
    df_agg = df.group_by("simulator", "controller", "wdir").agg(pl.col("Cp").sum())

    # Make normalizer column (Not sure how to this cleverly)
    normalizer_column = []
    for simulator, wdir in df_agg.select("simulator", "wdir").iter_rows():
        normalizer_column.append(
            df_agg.filter(simulator=simulator, wdir=wdir, controller="nocontrol")["Cp"][0]
        )
    df_agg = df_agg.with_columns(pl.Series(name="normalizer", values=normalizer_column))
    df_agg = df_agg.with_columns(P_norm=100 * (pl.col("Cp") / pl.col("normalizer") - 1))
    print(df_agg)

    # Plot
    g = sns.catplot(
        df_agg.filter(pl.col("controller") != "nocontrol"),
        kind="bar",
        x="controller",
        y="P_norm",
        col="wdir",
        hue="simulator",
        order=["thrustcontrol", "yawcontrol", "jointcontrol"],
    )
    g.set_axis_labels("", "Power increase (%)")
    plt.savefig(fig_dir / "LES_vs_MITwindfarm.png", dpi=300, bbox_inches="tight")
    plt.show()
