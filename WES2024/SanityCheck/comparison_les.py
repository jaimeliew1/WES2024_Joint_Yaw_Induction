from pathlib import Path

import altair as alt
import polars as pl


DATA_FNS = {
    "linear": Path(__file__).parent.parent.parent / "data_linear_superposition/LES_case_sanity_check.csv",
    "niayifar": Path(__file__).parent.parent.parent / "data_niayifar_superposition/LES_case_sanity_check.csv",
}
FIGDIR = Path("fig")
FIGDIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    df_list = []
    for key, fn in DATA_FNS.items():
        df = pl.read_csv(fn).with_columns(superposition=pl.lit(key))
        df_list.append(df)
    df = pl.concat(df_list)

    df_filt = df#.filter(wdir=0.0)
    df_pand = df_filt.to_pandas()
    print(df_filt)
    points = (
        alt.Chart(df_pand)
        .mark_point(filled=True, size=70, stroke="black")
        .encode(
            x=alt.X("x", axis=None).scale(domain=[-3, 37]),
            y=alt.Y("y", axis=None).scale(domain=[-3, 40]),
            color=alt.Color(
                "Cp",
                title="Cp",
                legend=alt.Legend(titleAnchor="middle"),
            ).scale(scheme="viridis", domainMax=0.6),
        )
    )

    text = (
        alt.Chart(df_pand)
        .mark_text(dy=-10)
        .encode(
            x=alt.X("x", axis=None),
            y=alt.Y("y", axis=None),
            text=alt.Text("Cp", format=",.2f"),
        )
    )

    chart = (
        (points + text)
        .properties(width=200, height=200)
        .facet(column="superposition")
    )
    chart.save(FIGDIR / "comparison_les.png", ppi=300)
