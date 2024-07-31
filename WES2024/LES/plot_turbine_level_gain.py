from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import polars as pl
from mitwindfarm import Square, Layout
from tqdm import tqdm


from WES2024.LES.run import extract_setpoints_from_json, normalize_by_upstream, ROW_INDICES

data_dir = Path(__file__).parent / "LES_output"
input_dir = Path(__file__).parent / "LES_input"
fig_dir = Path(__file__).parent / "fig"
fig_dir.mkdir(parents=True, exist_ok=True)

LAYOUT = Square(6.0, 5).rotate(45).rotate(-2.5)


def plot_text_on_layout(
    vals: list[float | str],
    ax: Optional[plt.Axes] = None,
    layout: Layout = LAYOUT,
    as_percent: bool = False,
    textbox: Optional[str] = None,
) -> None:

    if ax is None:
        plt.figure()
        plt.axis("equal")
        ax = plt.gca()

    if not isinstance(vals[0], str):
        cmap = plt.cm.RdYlGn
        norm = colors.Normalize(vmin=-np.max(np.abs(vals)), vmax=np.max(np.abs(vals)))
        _colors = cmap(norm(vals))
    else:
        _colors = ["0.5" for _ in vals]
    for idx, (x, y, val, _color) in enumerate(zip(layout.x, layout.y, vals, _colors)):
        ax.plot(x, y, ".", ms=10, c=_color)
        if isinstance(vals[0], str):
            ax.text(x, y, val, ha="center")
        elif as_percent:
            ax.text(x, y, f"{val*100:+2.1f}\%", ha="center", va="bottom")
        else:
            ax.text(x, y, f"{val:2.2f}", ha="center", va="bottom")

    if textbox:
        ax.text(
            0.02,
            0.02,
            textbox,
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
        )


@dataclass
class SimData:
    df: pl.DataFrame
    name: str

    @classmethod
    def from_files(cls, output_fn: Path, name: str = "") -> "SimData":
        _df = pl.read_csv(output_fn)
        Cp_upstream_norm = normalize_by_upstream(
            _df["Cp"].to_numpy(), row_indices=ROW_INDICES[-2.5]
        )

        df = _df.select(
            "Cp",
            pl.Series(name="Cp_upstream_norm", values=Cp_upstream_norm),
            "Ctprime",
            np.rad2deg(pl.col("yaw")),
        )

        return cls(df, name)

    @property
    def power(self):
        return self.df["Cp"].mean()

    def __repr__(self):
        return f"Cp: {self.power}\n{self.df}"

    def compare_with(self, other: "SimData") -> pl.DataFrame:
        return other.df.with_columns(
            pl.col("Cp") / self.df["Cp"] - 1,
            pl.col("Ctprime") / self.df["Ctprime"] - 1,
            pl.col("yaw") / self.df["yaw"] - 1,
            pl.col("Cp_upstream_norm") / self.df["Cp_upstream_norm"] - 1,
        )


def plot_compare(
    sim1: SimData,
    sim2: SimData,
    title: Optional[str] = None,
    save: Optional[Path] = None,
    upstream_norm: bool = False,
):
    if upstream_norm:
        Cp_key = "Cp_upstream_norm"
        power_title = "Upstream-normalized power"
    else:
        power_title = "$C_P$"
        Cp_key = "Cp"
    df_diff = sim1.compare_with(sim2)

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=1.0 * np.array([9, 9]))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plot_text_on_layout(
        sim1.df[Cp_key].to_numpy(), ax=axes[0, 0], textbox=f"$C_p=$ {sim1.power:.2f}"
    )
    plot_text_on_layout(
        sim2.df[Cp_key].to_numpy(), ax=axes[1, 0], textbox=f"$C_p=$ {sim2.power:.2f}"
    )
    plot_text_on_layout(
        df_diff[Cp_key].to_numpy(),
        ax=axes[2, 0],
        as_percent=True,
        textbox=f"$C_p ${100*(sim2.power/sim1.power - 1):+.2f}\%",
    )

    plot_text_on_layout(sim1.df["Ctprime"].to_numpy(), ax=axes[0, 1])
    plot_text_on_layout(sim2.df["Ctprime"].to_numpy(), ax=axes[1, 1])
    plot_text_on_layout(df_diff["Ctprime"].to_numpy(), ax=axes[2, 1], as_percent=True)

    plot_text_on_layout(sim1.df["yaw"].to_numpy(), ax=axes[0, 2])
    plot_text_on_layout(sim2.df["yaw"].to_numpy(), ax=axes[1, 2])
    plot_text_on_layout(df_diff["yaw"].to_numpy(), ax=axes[2, 2], as_percent=True)

    # titles
    axes[0, 0].set_title(power_title)
    axes[0, 1].set_title("$C_T'$ setpoint [-]")
    axes[0, 2].set_title("yaw setpoint [deg]")

    axes[0, 0].set_ylabel(sim1.name)
    axes[1, 0].set_ylabel(sim2.name)
    axes[2, 0].set_ylabel(rf"{sim1.name}$\rightarrow$\\{sim2.name}")
    if title:
        plt.suptitle(title)
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    sims = [
        SimData.from_files(
            data_dir / "LES_wdir-2.5_nocontrol.csv",
            name="LES_nocontrol",
        ),
        SimData.from_files(
            data_dir / "LES_wdir-2.5_thrustcontrolcalibration.csv",
            name="LES_thrustcontrolcalibration",
        ),
        SimData.from_files(
            data_dir / "LES_wdir-2.5_yawcontrol.csv",
            name="LES_yawcontrol",
        ),
        SimData.from_files(
            data_dir / "LES_wdir-2.5_jointcontrol.csv",
            name="LES_jointcontrol",
        ),
        SimData.from_files(
            data_dir / "LES_wdir-2.5_jointunicontrol.csv",
            name="LES_jointunicontrol",
        ),
        SimData.from_files(
            data_dir / "MITWindfarm_wdir-2.5_nocontrol.csv",
            name="MIT_nocontrol",
        ),
        SimData.from_files(
            data_dir / "MITWindfarm_wdir-2.5_thrustcontrol.csv",
            name="MIT_thrustcontrol",
        ),
        SimData.from_files(
            data_dir / "MITWindfarm_wdir-2.5_thrustcontrolcalibration.csv",
            name="MIT_thrustcontrolcalibration",
        ),
        SimData.from_files(
            data_dir / "MITWindfarm_wdir-2.5_jointcontrol.csv",
            name="MIT_jointcontrol",
        ),
        SimData.from_files(
            data_dir / "MITWindfarm_wdir-2.5_yawcontrol.csv",
            name="MIT_yawcontrol",
        ),
    ]
    simdata = {x.name: x for x in sims}
    # print names:
    print("Available simulation results:")
    for key in simdata:
        print(key)

    pairs_to_compare = [
        # MIT power gains
        ("MIT_nocontrol", "MIT_jointcontrol", False),
        ("MIT_nocontrol", "MIT_yawcontrol", False),
        ("MIT_nocontrol", "MIT_thrustcontrol", False),
        # New thrustcontrol with calibration thrust control
        ("MIT_thrustcontrol", "MIT_thrustcontrolcalibration", False),
        # LES power gains
        # ("LES_nocontrol", "LES_thrustcontrol", False),
        ("LES_nocontrol", "LES_thrustcontrolcalibration", False),
        ("LES_nocontrol", "LES_yawcontrol", False),
        ("LES_nocontrol", "LES_jointcontrol", False),
        ("LES_nocontrol", "LES_jointunicontrol", False),
        # new thrust control vs calibration thrust control
        # ("LES_thrustcontrol", "LES_thrustcontrolcalibration", False),
        # MIT vs LES
        ("LES_nocontrol", "MIT_nocontrol", True),
        ("LES_thrustcontrolcalibration", "MIT_thrustcontrolcalibration", True),
        ("LES_yawcontrol", "MIT_yawcontrol", True),
        ("LES_jointcontrol", "MIT_jointcontrol", True),
        ("LES_jointunicontrol", "MIT_jointcontrol", True),
    ]

    for sim1, sim2, upstream_norm in tqdm(pairs_to_compare):
        title = f"{sim1}->{sim2}"
        plot_compare(
            simdata[sim1],
            simdata[sim2],
            title=title,
            upstream_norm=upstream_norm,
            save=fig_dir / f"compare_{title}.png",
        )
