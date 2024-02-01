from typing import Callable, Any
from MITRotor.BEM import BEM, BEMSolution
from MITRotor.Geometry import BEMGeometry
from MITRotor.ReferenceTurbines import IEA15MW, RotorDefinition
from dualitic import DualNumber, DualVariables
from numpy.typing import ArrayLike
import numpy as np
from rich import print

from UnifiedMomentumModel.Utilities.FixedPointIteration import FixedPointIterationCompatible, FixedPointIterationResult

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


def undual(x):
    """
    'unduals' a number returning the real part if the number is a DualNumber, or
    just returns the value again if it is not.
    """
    if isinstance(x, DualNumber):
        return x.real[0]
    return x


@dualfixedpointiteration()
class DualBEM:
    def __init__(self, rotor: RotorDefinition, geometry: BEMGeometry = None):
        self.bem = BEM(rotor, geometry)
        self._niter_primal = None

    def sample_points(self, yaw: float = 0.0) -> tuple:
        return self.bem.sample_points(yaw)

    def initial_guess(self, pitch, tsr, yaw, U=1.0, wdir=0.0):
        # sol = self.bem(undual(pitch), undual(tsr), undual(yaw))
        # self._niter_primal = sol.niter
        # return sol.a(grid="radial"), sol.aprime(grid="radial")
        return self.bem.initial_guess(undual(pitch), undual(tsr), undual(yaw), undual(U), undual(wdir))

    def residual(self, x, pitch, tsr, yaw, U=1.0, wdir=0.0):
        res = self.bem.residual(x, pitch, tsr, yaw, U, wdir)
        return res

    def post_process(self, result: FixedPointIterationResult, pitch, tsr, yaw, U=1.0, wdir=0.0) -> BEMSolution:
        result.niter = (self._niter_primal, result.niter)
        return self.bem.post_process(result, pitch, tsr, yaw, U, wdir)


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
