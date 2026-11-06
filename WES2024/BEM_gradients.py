from dataclasses import dataclass, fields
from typing import Callable, Any
from MITRotor import BEM, BEMSolution, BEMGeometry, IEA15MW, RotorDefinition
from MITRotor.BEMSolver import average
from MITRotor.Geometry import expand_to_Np

from dualitic import DualNumber, DualVariables
from numpy.typing import ArrayLike
import numpy as np
from rich import print
from scipy.integrate import trapezoid

from UnifiedMomentumModel.Utilities.FixedPointIteration import (
    FixedPointIterationCompatible,
    FixedPointIterationResult,
)

PITCH, TSR, YAW = np.deg2rad(-8.5), 10, np.deg2rad(10)
params = {
    # "ref": (PITCH, TSR, YAW),
    # "dual_pitch": (DualNumber(PITCH, [1]), TSR, YAW),
    # "dual_tsr": (PITCH, DualNumber(TSR, [1]), YAW),
    # "dual_yaw": (PITCH, TSR, DualNumber(YAW, [1])),
    "dual_all": DualVariables([PITCH, TSR, YAW, 0.5, 0.0]),
}

geom = BEMGeometry(3, 4)


def _dualfixedpointiteration(
    f: Callable[[ArrayLike, Any], np.ndarray],
    x0: np.ndarray,
    args=(),
    kwargs={},
    eps=0.00001,
    maxiter=100,
    relax=0,
    callback=None,
) -> FixedPointIterationResult:
    for c in range(maxiter):
        residuals = f(x0, *args, **kwargs)
        x0 = [_x0 + (1 - relax) * _r for _x0, _r in zip(x0, residuals)]
        residuals_real = [_r.real if isinstance(_r, DualNumber) else _r for _r in residuals]
        residuals_dual = [_r.dual if isinstance(_r, DualNumber) else [0] for _r in residuals]

        max_resid_real = [np.nanmax(np.abs(_r)) for _r in residuals_real]
        max_resid_dual = [np.nanmax(np.abs(_r)) for _r in residuals_dual]

        max_max_resid = np.nanmax([max_resid_real, max_resid_dual])
        if callback:
            callback(x0)

        if max_max_resid < eps:
            converged = True
            break
    else:
        converged = False

    if maxiter == 0:
        return FixedPointIterationResult(False, 0, np.nan, np.nan, x0)
    return FixedPointIterationResult(converged, c, relax, max_max_resid, x0)


def dualfixedpointiteration(
    max_iter: int = 100, tolerance: float = 1e-6, relaxation: float = 0.0
) -> FixedPointIterationCompatible:
    def decorator(cls: FixedPointIterationCompatible) -> Callable:
        def call(self, *args, **kwargs):
            if hasattr(self, "pre_process"):
                self.pre_process(*args, **kwargs)

            callback = self.callback if hasattr(self, "callback") else None

            x0 = self.initial_guess(*args, **kwargs)
            result = _dualfixedpointiteration(
                self.residual,
                x0,
                args=args,
                kwargs=kwargs,
                eps=tolerance,
                maxiter=max_iter,
                relax=relaxation,
                callback=callback,
            )

            if hasattr(self, "post_process"):
                return self.post_process(result, *args, **kwargs)
            else:
                return result

        setattr(cls, "__call__", call)
        return cls

    return decorator

def dualadaptivefixedpointiteration(
    max_iter: int = 100, tolerance: float = 1e-6, relaxations: list[float] = [0.0]
) -> FixedPointIterationCompatible:
    def decorator(cls: FixedPointIterationCompatible) -> Callable:
            def call(self, *args, **kwargs):
                if hasattr(self, "pre_process"):
                    self.pre_process(*args, **kwargs)
                callback = self.callback if hasattr(self, "callback") else None

                for relaxation in relaxations:
                    x0 = self.initial_guess(*args, **kwargs)
                    result = _dualfixedpointiteration(
                        self.residual,
                        x0,
                        args=args,
                        kwargs=kwargs,
                        eps=tolerance,
                        maxiter=max_iter,
                        relax=relaxation,
                        callback=callback,
                    )
                    if result.converged:
                        break

                if hasattr(self, "post_process"):
                    return self.post_process(result, *args, **kwargs)
                else:
                    return result

            setattr(cls, "__call__", call)
            return cls

    return decorator


def undual(x):
    """
    'unduals' a number returning the real part if the number is a DualNumber, or
    just returns the value again if it is not.
    """
    if isinstance(x, DualNumber):
        return x.real[0]
    return x


class DualBEMGeometry(BEMGeometry):
    """BEMGeometry whose averages keep dual parts (MITRotor's versions pass through np.asarray)."""

    def annulus_average(self, X):
        return (1 / (2 * np.pi)) * trapezoid(X, self.theta, axis=-1)

    def rotor_average(self, X):
        return 2 * trapezoid(X * self.mu, self.mu, axis=-1)


@dataclass
class DualBEMSolution(BEMSolution):
    """
    BEMSolution whose power coefficients keep dual parts (MITRotor's versions call np.asarray(tsr)).
    Also carries the near-wake length x0, which the curled wake models read from `rotor_solution.extra`. This is the
    TI-free LUT value; WES2024.CustomRotors.BEMTI overwrites it with the TI-dependent x0.
    """

    x0: float = np.inf

    def Cp(self, grid="rotor"):
        return average(self.geom, self.tsr * self.geom.mu_mesh * self.Ctau_uncorr(grid="sector"), grid)

    def Cp_corr(self, grid="rotor"):
        return average(self.geom, self.tsr * self.geom.mu_mesh * self.Ctau(grid="sector"), grid)


# Relaxations: the fixed-point iteration (aero_props -> an, a') is unstable for relax <~ 0.4 at nonzero yaw (a failed stage
# wastes max_iter iterations); relax 0.5 converges in ~22 iterations at every set point tried (pitch -10..10 deg,
# TSR 3..10, yaw 0..45 deg), 0.8 in ~66 and 0.96 in ~360. The old schedule [0.25, 0.5, 0.96] spent 500 iterations on
# the failing 0.25 stage for any yaw != 0.
@dualadaptivefixedpointiteration(max_iter=500, relaxations=[0.5, 0.8, 0.96])
class DualBEM:
    def __init__(self, 
                 rotor: RotorDefinition, 
                 geometry: BEMGeometry = None, 
                 momentum_model=None,
                 aerodynamic_model=None,
                 tangential_induction_model=None):
        # Nr = 128 gives ~0.1% error in Cp, Ct, Cq (R1/plot_bem_convergence.py); the old Nr = 10 was 4-6% low.
        # Ntheta converges spectrally: 12 gives ~1e-9.
        geometry = geometry or BEMGeometry(Nr=128, Ntheta=12)
        if not isinstance(geometry, DualBEMGeometry):
            geometry = DualBEMGeometry(geometry.Nr, geometry.Ntheta)
        self.bem = BEM(rotor, 
                       geometry, 
                       momentum_model=momentum_model, 
                       aerodynamic_model=aerodynamic_model,
                       tangential_induction_model=tangential_induction_model)
        self._niter_primal = None
        self.geometry = self.bem.geometry

    def sample_points(self, yaw: float = 0.0, tilt: float = 0.0) -> tuple:
        return self.bem.sample_points(yaw, tilt)

    def pre_process(self, pitch, tsr, yaw=0.0, U=1.0, wdir=0.0, tilt=0.0):
        """
        Dual-number counterpart of BEM.pre_process for scalar setpoints. The effective yaw is kept as
        the (possibly dual) yaw so that gradients propagate; tilt is not supported.
        """
        if np.any(np.asarray(undual(tilt)) != 0):
            raise NotImplementedError("DualBEM does not support tilt.")
        self.bem.scalar_inputs = True
        self.bem.aerodynamic_model.eff_yaw = yaw
        self.bem.aerodynamic_model.delta_theta = 0.0
        geom = self.bem.geometry
        geom.mu_mesh = expand_to_Np(geom.mu_mesh)
        geom.theta_mesh = expand_to_Np(geom.theta_mesh)

    def initial_guess(self, pitch, tsr, yaw=0.0, U=1.0, wdir=0.0, tilt=0.0):
        return self.bem.initial_guess(
            undual(pitch), undual(tsr), undual(yaw), undual(U), undual(wdir)
        )

    def residual(self, x, pitch, tsr, yaw=0.0, U=1.0, wdir=0.0, tilt=0.0):
        res = self.bem.residual(x, pitch, tsr, yaw, U, wdir, tilt)
        return res

    def post_process(
        self, result: FixedPointIterationResult, pitch, tsr, yaw=0.0, U=1.0, wdir=0.0, tilt=0.0
    ) -> BEMSolution:
        result.niter = (self._niter_primal, result.niter)
        sol = self.bem.post_process(result, pitch, tsr, yaw, U, wdir, tilt)
        kwargs = {f.name: getattr(sol, f.name) for f in fields(sol)}
        momentum = self.bem.momentum_model
        if hasattr(momentum, "compute_x0"):
            kwargs["x0"] = momentum.compute_x0(average(sol.geom, sol.aero_props.C_x), yaw)
        return DualBEMSolution(**kwargs)


if __name__ == "__main__":
    bem = BEM(IEA15MW(), geom)
    dual_bem = DualBEM(IEA15MW(), geom)
    # print(bem(0, 7, 0).Cp())

    guess = bem.initial_guess(0, 7, 0)

    results = {}

    for label, param in params.items():
        # _a, _aprime = bem.residual(guess, *param)
        # x0 = dual_bem.inital_guess(*param)
        # res = dual_bem.residual(x0, *param)
        # print(label, res)

        # results[label] = (_a, _aprime)
        sol = dual_bem(*param)
        print(label, sol)
        print(label, sol.Cp())

    # for label, (a, aprime) in results.items():
    #     print(label, a)
