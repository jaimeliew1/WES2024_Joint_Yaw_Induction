"""
Constrained controllers for the R1 BEM runs.

ConstrainedMixin replaces Controller.optimise with an SLSQP solve (one AD farm evaluation per iterate, shared by
the objective and all constraints) when either of these applies:

- `Komega_constraint`: equality K-omega^2 law per rotor, Cp_local / tsr^3 = CP_OPT / TSR_OPT^3 (tsr is defined
  with the rotor-effective wind speed, so the local Cp = rotor.Cp / REWS^3 must be used. optimise.py's
  K_omega_constraint_func uses the freestream-normalised rotor.Cp, which differs for waked rotors).
- `Ct_max`: inequality Ct_local <= Ct_max for every rotor. Needed by the BP2016 Gaussian wake, which is undefined
  (sqrt(1 - Ct)) for Ct >= 1.

YawCapMixin limits |yaw| to YAW_MAX (30 deg) for every R1 control strategy; optimise.py (used by the original
WES2024 scripts) keeps its +-45 / +-50 deg bounds.
"""

import numpy as np
from numpy.typing import ArrayLike
from dualitic import DualVariables
from scipy.optimize import minimize

import sys

from WES2024 import optimise
from WES2024.CustomRotors import BEMUnifiedMomentumLUTAnnulus
from WES2024.optimise import (
    CP_OPT,
    PITCH_OPT,
    TSR_OPT,
    CTPRIME_OPT,
    JointControl,
    JointControlBEM,
    NoControlBEM,
    ThrustControlBEM,
    YawControlBEM,
    YawControlModBEM,
    YawControlKOmegaBEM,
)

# SLSQP stops when the objective changes by less than ftol. It must stay above the numerical noise in Cp (~1e-6 to
# 1e-5 for TANDEM's adaptive integrator); below that SLSQP cannot satisfy its test and wastes line-search evaluations.
SLSQP_FTOL = 1e-6
YAW0_DEFAULT = (
    1e-2  # small yaw offset to escape the symmetric saddle at yaw=0 (sign picks the yaw direction)
)
YAW_MAX = np.deg2rad(30.0)  # |yaw| cap for all R1 control strategies
CTPRIME_MIN = 1e-3  # TANDEM / BP cannot handle Ct' -> 0 (see tmp/extra/generate_two_turbine_AD_wakemodels.py)


class YawCapMixin:
    """
    Caps |yaw| at YAW_MAX. `yaw_block` is the index of the block of N variables that holds the yaw angles
    (0: yaw only, 1: [x, yaw] as in K-omega control and the AD joint control, 2: [pitch, TSR, yaw]);
    None for strategies without a yaw variable. Place this class before the optimise.py controller in the bases.
    """

    yaw_block = None

    def bounds(self):
        bounds = list(super().bounds())
        if self.yaw_block is None:
            return bounds
        if len(bounds) == 1:  # one tuple shared by all variables (AD YawControl)
            lo, hi = bounds[0]
            return [(max(lo, -YAW_MAX), min(hi, YAW_MAX))]
        for i in range(self.yaw_block * self.N, (self.yaw_block + 1) * self.N):
            lo, hi = bounds[i]
            bounds[i] = (max(lo, -YAW_MAX), min(hi, YAW_MAX))
        return bounds


def _val_grad(v, n):
    """(value, gradient) of a possibly-dual scalar."""
    if hasattr(v, "dual"):
        return float(np.ravel(v.real)[0]), np.asarray(v.dual).reshape(-1, n)[0]
    return float(v), np.zeros(n)


class ConstrainedMixin(YawCapMixin):
    Ct_max = None
    yaw0 = YAW0_DEFAULT  # initial yaw guess [rad]; set per wake model by the generate script
    # SLSQP stopping tolerance on the objective; lower it only for noise-free models (see SLSQP_FTOL above)
    ftol = SLSQP_FTOL
    # Optional typical size of each block of N variables (e.g. (pitch, TSR, yaw) = (0.05, 1, 0.3) for joint control).
    # With it SLSQP works on z = (x - x0) / scale, so that all variables are O(1); without it on x itself.
    var_scales = None
    maxiter = 100

    def _evaluate(self, x):
        key = np.asarray(x, dtype=float).tobytes()
        if getattr(self, "_cache_key", None) != key:
            n = len(x)
            sol = self.solve_for_setpoints(DualVariables(x))
            cp, cp_grad = _val_grad(sol.Cp, n)
            ct, ct_jac, kw, kw_jac = [], [], [], []
            for r in sol.rotors:
                if self.Ct_max is not None:
                    v, g = _val_grad(r.Ct / r.REWS**2, n)
                    ct.append(self.Ct_max - v)
                    ct_jac.append(-g)
                if self.Komega_constraint:
                    v, g = _val_grad(r.Cp / r.REWS**3 / r.extra.tsr**3 - CP_OPT / TSR_OPT**3, n)
                    kw.append(v)
                    kw_jac.append(g)
            self._cache = dict(
                f=-cp,
                g=-cp_grad,
                ct=np.array(ct),
                ct_jac=np.array(ct_jac),
                kw=np.array(kw),
                kw_jac=np.array(kw_jac),
            )
            self._cache_key = key
        return self._cache

    # FD step size in scaled z-space (z is O(1) by construction of var_scales), used only when optimise() is
    # called with use_gradients=False. Must stay well above the noise floor of the underlying solve (TANDEM's
    # adaptive integrator: ~1e-6 to 1e-5 in Cp, see SLSQP_FTOL above), or the finite-difference gradient is noise.
    fd_eps = 1e-3

    def _evaluate_fd(self, x):
        key = np.asarray(x, dtype=float).tobytes()
        if getattr(self, "_cache_key_fd", None) != key:
            sol = self.solve_for_setpoints(np.asarray(x, dtype=float))
            ct, kw = [], []
            for r in sol.rotors:
                if self.Ct_max is not None:
                    ct.append(self.Ct_max - float(r.Ct) / float(r.REWS) ** 2)
                if self.Komega_constraint:
                    kw.append(float(r.Cp) / float(r.REWS) ** 3 / float(r.extra.tsr) ** 3 - CP_OPT / TSR_OPT**3)
            self._cache_fd = dict(f=-float(sol.Cp), ct=np.array(ct), kw=np.array(kw))
            self._cache_key_fd = key
        return self._cache_fd

    def optimise(self, verbose=False, use_gradients=True, Cp_constraint=None):
        if not (self.Komega_constraint or self.Ct_max is not None):
            return super().optimise(
                verbose=verbose, use_gradients=use_gradients, Cp_constraint=Cp_constraint
            )
        assert Cp_constraint is None

        N = len(self.initial_guess())
        bounds = self.bounds()
        if len(bounds) == 1:  # YawControl-style single bound tuple shared by all variables
            bounds = bounds * N
        x0 = np.asarray(self.initial_guess(), dtype=float)
        if self.var_scales is None:  # SLSQP on x itself
            x_ref, scale = np.zeros(N), np.ones(N)
        else:  # SLSQP on z = (x - x0) / scale
            x_ref, scale = x0, np.repeat(self.var_scales, N // len(self.var_scales))
        to_x = lambda z: x_ref + scale * z
        evaluate = self._evaluate if use_gradients else self._evaluate_fd

        constraints = []
        if self.Ct_max is not None:
            constraints.append(
                dict(
                    type="ineq",
                    fun=lambda z: evaluate(to_x(z))["ct"],
                    **(dict(jac=lambda z: evaluate(to_x(z))["ct_jac"] * scale) if use_gradients else {}),
                )
            )
        if self.Komega_constraint:
            constraints.append(
                dict(
                    type="eq",
                    fun=lambda z: evaluate(to_x(z))["kw"],
                    **(dict(jac=lambda z: evaluate(to_x(z))["kw_jac"] * scale) if use_gradients else {}),
                )
            )
        options = dict(ftol=self.ftol, maxiter=self.maxiter)
        if not use_gradients:
            options["eps"] = self.fd_eps
        res = minimize(
            lambda z: evaluate(to_x(z))["f"],
            (x0 - x_ref) / scale,
            jac=(lambda z: evaluate(to_x(z))["g"] * scale) if use_gradients else None,
            bounds=[
                ((lo - r) / sc, (hi - r) / sc) for (lo, hi), r, sc in zip(bounds, x_ref, scale)
            ],
            constraints=constraints,
            method="SLSQP",
            options=options,
        )
        res.x = to_x(res.x)
        self.result = res
        if verbose:
            print(res)
        return self.solve_for_setpoints(res.x)


class YawControlR1(ConstrainedMixin, YawControlBEM):
    """Yaw only, pitch and TSR fixed at the single-turbine optimum."""

    yaw_block = 0

    def initial_guess(self) -> ArrayLike:
        return [self.yaw0] * self.N


class YawModControlR1(ConstrainedMixin, YawControlModBEM):
    """Yaw only, TSR = TSR_OPT cos(yaw) so that lambda~ = Omega R / (U cos yaw) is constant (Li & Yang 2021)."""

    yaw_block = 0

    def initial_guess(self) -> ArrayLike:
        return [self.yaw0] * self.N


class YawKOmegaControlR1(ConstrainedMixin, YawControlKOmegaBEM):
    """Yaw plus TSR, with TSR set by the K-omega^2 torque law (pitch fixed)."""

    yaw_block = 1

    def initial_guess(self) -> ArrayLike:
        return [TSR_OPT] * self.N + [self.yaw0] * self.N


class ThrustControlR1(ConstrainedMixin, ThrustControlBEM):
    pass


class JointControlR1(ConstrainedMixin, JointControlBEM):
    yaw_block = 2

    def initial_guess(self) -> ArrayLike:
        return [PITCH_OPT] * self.N + [TSR_OPT] * self.N + [self.yaw0] * self.N


class NoControlR1(NoControlBEM):
    pass


class JointAD(ConstrainedMixin, JointControl):
    """Joint (Ct', yaw) control of an actuator-disk rotor, SLSQP with scaled variables and AD gradients."""

    yaw_block = 1

    def initial_guess(self):
        return [CTPRIME_OPT] * self.N + [self.yaw0] * self.N

    def bounds(self):
        return [(max(lo, CTPRIME_MIN), hi) if i < self.N else (lo, hi) for i, (lo, hi) in enumerate(super().bounds())]


# control strategy -> controller, the R1 strategies of the two-turbine and farm figures
METHODS = {
    "NoControl": NoControlR1,
    "ThrustControl": ThrustControlR1,  # pitch and TSR
    "YawControl": YawControlR1,  # pitch and TSR fixed at the single-turbine optimum
    "YawModControl": YawModControlR1,  # TSR = TSR_OPT cos(yaw): constant modified TSR (Li & Yang 2021), pitch fixed
    "YawKOmegaControl": YawKOmegaControlR1,  # TSR follows the K-omega^2 torque law, pitch fixed
    "JointControl": JointControlR1,  # pitch, TSR and yaw
}

# Typical size of each block of N variables (e.g. (pitch [rad], TSR, yaw [rad]) for joint control), see var_scales.
VAR_SCALES = {
    "ThrustControl": (0.05, 1.0),
    "YawControl": (0.3,),
    "YawModControl": (0.3,),
    "YawKOmegaControl": (1.0, 0.3),
    "JointControl": (0.05, 1.0, 0.3),
}

# Ct limit per wake model. BP2016 (and the axisymmetric Gaussian, Ct cos(yaw) < 8 sigma_0^2 = 1) are undefined for Ct >= 1,
# hence 0.95. For the other models the limit does not bind (BEM Ct < 1.5); it only switches on the scaled, ftol-controlled
# SLSQP path of ConstrainedMixin.
CT_MAX = {"gaussBP": 0.95, "gauss": 0.95}
CT_MAX_DEFAULT = 1.5


def use_annulus_optimum():
    """
    Re-point the PITCH_OPT / TSR_OPT / CP_OPT of WES2024.optimise (read by its Controller classes, e.g. NoControlBEM uses them
    as the literal unyawed set point) and of this module (read by the mixins, e.g. the K-omega^2 constraint) at the
    single-turbine optimum of the annulus-closure BEM LUT rotor, which every paper figure uses (wakemodels.get_bem_lut_rotor).
    The constants in optimise.py belong to get_bem_rotor's "rotor_induction_tiploss" closure. Rebinds names in this process
    only, so call it at the import of the generator script (foreach's spawned workers re-import that script).
    """
    for mod in (optimise, sys.modules[__name__]):
        mod.PITCH_OPT = BEMUnifiedMomentumLUTAnnulus.PITCH_OPT
        mod.TSR_OPT = BEMUnifiedMomentumLUTAnnulus.TSR_OPT
        mod.CP_OPT = BEMUnifiedMomentumLUTAnnulus.CP_OPT


def make_controller(method, layout, windfarm, model, ftol, maxiter, yaw0=None, x0=None):
    """Controller `method` (key of METHODS) with the SLSQP settings of wake model `model`, started from yaw0 or the vector x0."""
    controller = METHODS[method](layout, windfarm)
    if method == "NoControl":
        return controller
    controller.Ct_max = CT_MAX.get(model, CT_MAX_DEFAULT)
    controller.var_scales = VAR_SCALES[method]
    controller.ftol, controller.maxiter = ftol, maxiter
    if yaw0 is not None:
        controller.yaw0 = yaw0
    if x0 is not None:
        controller.initial_guess = lambda: list(x0)
    return controller
