from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Use Latex Fonts
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)

# IEA15MW turbine parameters
density = 1.225  # ?
P_rated = 15000000
R = 240  # ?


def main():
    Uinf = np.linspace(0, 25, 100)

    Cp_max = P_rated / (0.5 * density * np.pi * R**2 * Uinf**3)

    plt.figure()
    plt.plot(Uinf, Cp_max)
    plt.axhline(16 / 27, ls="--", c="r", lw=1)

    plt.xlabel(r"$U_\infty$")
    plt.ylabel(r"$C_P$ constraint")
    plt.ylim(0, 1)
    plt.xlim(0, Uinf.max())
    plt.savefig(FIGDIR / "above_rated_Cp.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
