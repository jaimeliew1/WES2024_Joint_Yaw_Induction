from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from foreach import foreach
from rich import print
from scipy.stats import norm

from WES2024 import utils
from WES2024.Generate import diamond_BEM_uncertainty

__all__ = ["generate"]

FILESTEM = Path(__file__).stem

stds = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
to_average = ["Cp", "Ct", "Ctprime", "an", "u4", "v4", "REWS"]
x_refined = np.linspace(-15, 15, 100)


# To do: add interpolation step
def _generate(x) -> pl.DataFrame:
    (turbine, method, wdir, _df), std = x

    if std == 0.0:
        averaged = {key: _df.filter(wdir_offset=0)[key][0] for key in to_average}
    else:
        pdf = norm(0, std).pdf(x_refined)
        averaged = {
            key: np.trapz(pdf * np.interp(x_refined, _df["wdir_offset"], _df[key]), x_refined)
            for key in to_average
        }

    out = dict(turbine=turbine, method=method, wdir=wdir, std=std) | averaged

    out = pl.from_dict(out)
    return out


@utils.cache_polars(utils.CACHEDIR / f"{FILESTEM}.csv")
def generate(regenerate=False):
    df = diamond_BEM_uncertainty.generate().select(
        "turbine", "method", "wdir", "wdir_offset", "Cp", "Ct", "Ctprime", "an", "u4", "v4", "REWS"
    )

    params_partial = (
        (turbine, method, wdir, _df)
        for (turbine, method, wdir), _df in df.group_by("turbine", "method", "wdir")
    )
    params = list(product(params_partial, stds))

    df_list = foreach(_generate, params, context="spawn", parallel=True)
    # breakpoint()
    # [print(x.schema) for x in df_list]
    df = pl.concat([x for x in df_list if x["method"].dtype == pl.Utf8])
    return df


if __name__ == "__main__":
    df = generate(regenerate=True)
    print(df)
