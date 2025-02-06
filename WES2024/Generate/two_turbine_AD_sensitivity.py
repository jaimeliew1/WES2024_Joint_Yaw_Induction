from pathlib import Path

import polars as pl
from dualitic import DualNumber
from foreach import foreach
from mitwindfarm import Windfarm, Niayifar, VariableKwGaussianWakeModel
from rich import print

from WES2024 import utils
from WES2024.CustomRotors import UnifiedLUTAD
from WES2024.Generate import two_turbine_AD

__all__ = ["generate"]

FILESTEM = Path(__file__).stem


# windfarm = Windfarm()

windfarm = Windfarm(
    rotor_model=UnifiedLUTAD(),
    superposition=Niayifar(),
    wake_model=VariableKwGaussianWakeModel(0.7683081169878619, 0.0, 0.004825109405157736, x0=3.0),
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
    df = two_turbine_AD.generate()
    params = [_df for _, _df in df.group_by("method", "wdir")]
    df = pl.concat(foreach(_generate, params, context="spawn", parallel=True))
    return df


if __name__ == "__main__":
    df = generate(regenerate=True)
    print(df)
