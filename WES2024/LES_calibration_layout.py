
import matplotlib.pyplot as plt
from mitwindfarm.Layout import Layout

from WES2024.Generate import LES_case_definitions


row_indices = [
    [24],
    [23, 19],
    [22, 18, 14],
    [21, 17, 13, 9],
    [20, 16, 12, 8, 4],
    [15, 11, 7, 3],
    [10, 6, 2],
    [5, 1],
    [0],
]

markers = [
    ".",
    "x",
    "^",
    "2",
    "+",
]
if __name__ == "__main__":
    df = (
        LES_case_definitions.generate()
        .filter(wdir=-2.5, controller="nocontrol")
        .select("turbine", "x", "y")
    )
    df = df
    layout = Layout(df["x"].to_numpy(), df["y"].to_numpy())

    plt.figure()
    plt.axis("equal")

    for i, row in enumerate(row_indices):
        color = plt.cm.tab10(i / 10)
        for upstreaminess, idx in enumerate(row):
            marker = markers[upstreaminess]
            plt.plot(layout.x[idx], layout.y[idx], marker=marker, ms=10, c=color)
            plt.text(layout.x[idx], layout.y[idx], f"{idx}")

    plt.savefig("asdf.png", dpi=500)
