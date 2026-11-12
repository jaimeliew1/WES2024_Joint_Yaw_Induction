"""
Rotor model read straight off the leading turbine of the two-turbine ABL LES sweep (LES_DATA: Ct' = 0.4..4.4,
yaw = 0..45 deg in 5 deg steps), i.e. the same actuator-disk physics as the 5x5 farm LES, with no cosine law and no BEM.

    Cp(Ct', |yaw|), Ct(Ct', |yaw|)   normalised by the case's rotor-equivalent wind speed REWS (as in
                                      R1.les_data.load_twoturbine_lead: Cp = P / (1/2 A REWS^3), Ct = Ct' (ud / REWS)^2)

Positive and negative yaw are assumed symmetric (|yaw|). Interpolation: PCHIP in |yaw| at every tabulated Ct', then linear in Ct'
(monotone, no overshoot; the table is noisy at the 1-2 % level, so no higher-order interpolation in Ct'). Outside the table
(|yaw| > 45 deg, Ct' outside 0.4..4.4) the value is clamped.

Wake side: induction, u4, v4, x0 are the Unified Momentum Model's (UnifiedLUTAD) for the same (Ct', yaw); the rotor only replaces Cp
and (replace_ct) Ct, which the wake models use for the vortex strength / deficit. The set points are (Ct', yaw), so this rotor is fed
the `Ctprime` and `yaw` columns of the optimiser CSV (the values the LES actually used); `setpoint_columns` tells
rowwise_cp_forward which columns to pass. Wakes are assumed not to change the rotor's induction, so REWS stays the UMM's.
"""

from dataclasses import replace

import numpy as np
from scipy.interpolate import PchipInterpolator

from WES2024.CustomRotors import UnifiedLUTAD
from WES2024.R1.les_data import load_twoturbine_lead


def load_table():
    """(ctp (n,), yaw_deg (m,), Cp (n, m), Ct (n, m)) of the leading turbine, ABL LES, REWS-normalised."""
    ctps = np.round(np.arange(0.4, 4.41, 0.4), 6)
    frames = [load_twoturbine_lead(c) for c in ctps]
    yaw = frames[0]["yaw"].to_numpy().astype(float)
    for f in frames:
        assert np.array_equal(f["yaw"].to_numpy(), frames[0]["yaw"].to_numpy())
    return ctps, yaw, np.stack([f["Cp_1"].to_numpy() for f in frames]), np.stack([f["Ct"].to_numpy() for f in frames])


class TabulatedLESAD(UnifiedLUTAD):
    """UnifiedLUTAD whose Cp (and optionally Ct) come from the tabulated two-turbine LES. Set points: (Ct', yaw [rad])."""

    setpoint_columns = ("Ctprime", "yaw")

    def __init__(self, rotor_grid=None, replace_ct: bool = True):
        super().__init__(rotor_grid)
        self.replace_ct = replace_ct
        self.ctp, self.yaw_deg, self._cp, self._ct = load_table()
        self._cp_y = [PchipInterpolator(self.yaw_deg, row) for row in self._cp]
        self._ct_y = [PchipInterpolator(self.yaw_deg, row) for row in self._ct]

    def table_values(self, Ctprime, yaw):
        """(Cp, Ct) REWS-normalised at Ct' and yaw [rad]."""
        g = min(np.degrees(abs(float(yaw))), self.yaw_deg[-1])
        cp = np.interp(float(Ctprime), self.ctp, [f(g) for f in self._cp_y])
        ct = np.interp(float(Ctprime), self.ctp, [f(g) for f in self._ct_y])
        return float(cp), float(ct)

    def __call__(self, x, y, z, windfield, Ctprime, yaw):
        sol = super().__call__(x, y, z, windfield, Ctprime, yaw)
        cp, ct = self.table_values(Ctprime, yaw)
        return replace(sol, Cp=cp * sol.REWS**3, Ct=ct * sol.REWS**2 if self.replace_ct else sol.Ct)
