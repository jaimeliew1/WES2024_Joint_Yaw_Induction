from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dualitic import DualNumber

from mitwindfarm.Layout import Layout
from mitwindfarm.windfarm import Windfarm, WindfarmSolution
from mitwindfarm.Wake import GaussianWakeModel


from plot_02_spiral import Spiral

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    # layout = Layout([0, 4, 8], [0, 0.5, 1.0])

    # x = DualVariables([2, 2, 2, 0, 0, 0])
    # setpoints = np.array(x).reshape([2, -1]).T
    sigma = DualNumber(0.25, [1.0])
    windfarm = Windfarm(wake_model=GaussianWakeModel(sigma=sigma))

    layout = Spiral(20, min_dist=4)
    setpoints = [(2.0, 0.0) for _ in layout]
    sol = windfarm(layout, setpoints)

    pad = 1
    xlim = (sol.layout.x.min() - pad, sol.layout.x.max() + 10)
    ylim = (sol.layout.y.min() - pad, sol.layout.y.max() + pad)

    # plot windfield and turbine stats
    _x, _y = np.linspace(*xlim, 400), np.linspace(*ylim, 401)
    xmesh, ymesh = np.meshgrid(_x, _y)

    wsp = sol.windfield.wsp(xmesh, ymesh, np.zeros_like(xmesh))

    _, axes = plt.subplots(2, 1)
    axes[0].imshow(wsp.real, extent=[*xlim, *ylim], vmin=0, vmax=2, origin="lower", cmap="RdYlBu_r")
    axes[1].imshow(wsp.dual, extent=[*xlim, *ylim], origin="lower", cmap="RdYlBu_r")

    for (turb_x, turb_y, _), rotor in zip(sol.layout, sol.rotors):
        # Draw turbine
        R = 0.5
        p = np.array([[0, 0], [+R, -R]])
        rotmat = np.array(
            [
                [np.cos(rotor.yaw), -np.sin(rotor.yaw)],
                [np.sin(rotor.yaw), np.cos(rotor.yaw)],
            ]
        )

        p = rotmat @ p + np.array([[turb_x], [turb_y]])

        axes[0].plot(p[0, :], p[1, :], "k", lw=2)
        axes[1].plot(p[0, :], p[1, :], "k", lw=2)
    axes[0].set(frame_on=False)
    axes[0].set(xticks=[], yticks=[])
    axes[1].set(frame_on=False)
    axes[1].set(xticks=[], yticks=[])

    plt.savefig(FIGDIR / "gradient_test.png", dpi=400, bbox_inches="tight")
