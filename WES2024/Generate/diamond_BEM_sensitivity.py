"""
NOTE: This script currently does not run (gradient issues)with wake added
turbulence. You may need to replace the line:

TI_out = np.sqrt(TI_base**2 + np.max(WATIs, axis=0) ** 2)

With TI_out = np.zeros_like(wsp_out) to get it to work for now.
"""
from pathlib import Path

import polars as pl
from dualitic import DualNumber
from foreach import foreach
from MITRotor.ReferenceTurbines import IEA15MW
from mitwindfarm import BEM, Windfarm, Niayifar, VariableKwGaussianWakeModel
from rich import print

from WES2024 import utils
from WES2024.BEM_gradients import DualBEM
from WES2024.CustomRotors import BEMUnifiedMomentumLUT
from WES2024.Generate import diamond_BEM

__all__ = ["generate"]

FILESTEM = Path(__file__).stem

PARALLEL = True
REGENERATE = True

windfarm = Windfarm(
    rotor_model=BEM(IEA15MW(), BEM_model=DualBEM, momentum_model=BEMUnifiedMomentumLUT()),
    superposition=Niayifar(),
    wake_model=VariableKwGaussianWakeModel(0.636, 0.0, 0.0, x0=1.0),
    TIamb=0.056,
)


def _generate(x) -> pl.DataFrame:
    _df = x
    windfarm_sol = utils.from_polars(_df, windfarm)
    rotate = DualNumber([0.0], [[1.0]])
    layout = windfarm_sol.layout.rotate(rotate)

    sol_grad = windfarm(layout, windfarm_sol.setpoints)

    dCpdwdir = [x.Cp.dual[0, 0] if isinstance(x.Cp, DualNumber) else 0.0 for x in sol_grad.rotors]
    turbine = list(range(len(_df)))

    df_sens = pl.DataFrame(dict(turbine=turbine, dCpdwdir=dCpdwdir))
    return _df.join(df_sens, on="turbine")


@utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False):
    df = diamond_BEM.generate()
    params = [_df for _, _df in df.group_by("method", "wdir")]
    df = pl.concat(foreach(_generate, params, context="spawn", parallel=PARALLEL))
    return df


if __name__ == "__main__":
    df = generate(regenerate=REGENERATE)
    print(df)
