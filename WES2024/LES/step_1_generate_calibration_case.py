from pathlib import Path
from typing import Optional

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from mitwindfarm import (
    Layout,
    VariableKwGaussianWakeModel,
    Windfarm,
    Niayifar,
)
from rich import print

from WES2024.CustomRotors import UnifiedLUTAD
from WES2024.LES.shared import STEP_1_DIR, TIAMB, SimulationDefinition, CALIBRATION_LAYOUT

#### PARAMETERS


SETPOINT_FN = STEP_1_DIR / "calibration_18x3.json"
FIG_FN = STEP_1_DIR / "calibration_case_setpoints.png"


# Setpoints
_Ctprimes = sum([[x, 2.0, 2.0] for x in np.linspace(1, 4, 9)], []) + sum(
    [[x, 1.0, 2.0] for x in np.linspace(1, 4, 9)], []
)
SETPOINTS = [(ctp, 0.0) for ctp in _Ctprimes]


def plot_text_on_layout(
    vals: list[float | str],
    layout: Layout,
    ax: Optional[plt.Axes] = None,
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


def plot_calibration_case(sim_case: SimulationDefinition, save_fn: Path) -> None:
    plt.figure()
    ax = plt.gca()

    Ctprimes = [x.ctp for x in sim_case.turbines]
    plot_text_on_layout(Ctprimes, sim_case.layout(), ax)
    ax.axis("equal")

    plt.savefig(save_fn, dpi=300, bbox_inches="tight")
    plt.close()


def run(fig_fn: Path, les_input_fn: Path) -> None:
    # Run model
    wakemodel = VariableKwGaussianWakeModel(0, 0, 0.07)
    windfarm = Windfarm(
        rotor_model=UnifiedLUTAD(), wake_model=wakemodel, TIamb=TIAMB, superposition=Niayifar()
    )

    sol = windfarm(CALIBRATION_LAYOUT, SETPOINTS)
    sim_case = SimulationDefinition.from_windfarmsolution(
        sol,
        wdir=-2.5,
        controller="calibration",
        casename="calibration_18x3",
        case_note="attempt_1",
    )

    sim_case.write_json(les_input_fn)

    plot_calibration_case(sim_case, fig_fn)


if __name__ == "__main__":
    run(FIG_FN, SETPOINT_FN)
