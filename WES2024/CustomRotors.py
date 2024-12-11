from abc import ABC, abstractmethod
import itertools
from pathlib import Path
from typing import Literal

import MITRotor
import numpy as np
import polars as pl
from foreach import foreach
from mitwindfarm import Area, Line, Point, RotorSolution
from mitwindfarm.Rotor import Rotor
from mitwindfarm.RotorGrid import RotorGrid
from mitwindfarm.Windfield import Windfield
from numpy.typing import ArrayLike
from scipy.interpolate import RectBivariateSpline
from UnifiedMomentumModel.Momentum import (
    MomentumBase,
    MomentumSolution,
    UnifiedMomentum,
    ThrustBasedUnified,
    LimitedHeck
)

model_Ctprime = UnifiedMomentum()
model_Ct = ThrustBasedUnified()

CACHE_FN_CTPRIME = Path(__file__).parent / "unified_momentum_model_Ctprime_table.csv"
CACHE_FN_CT = Path(__file__).parent / "unified_momentum_model_Ct_table.csv"


class CachedLUT(ABC):
    """
    Abstract base class for a Cached Look-Up Table (LUT) with interpolation capabilities.

    Attributes:
    ----------
    key1 : str
        First key used for table indexing.
    key2 : str
        Second key used for table indexing.
    to_interp : list[str]
        List of column names in the table to perform interpolation on.
    cache_fn : Path
        Path to the CSV file used for caching the table data.
    regenerate : bool, optional
        Flag indicating whether to regenerate the table even if cache file exists (default is False).
    s : float, optional
        Smoothing factor used in interpolation (default is 0.001).

    Methods:
    --------
    load() -> pl.DataFrame:
        Load the cached table data from the CSV file.

    save() -> pl.DataFrame:
        Save the current table data to the CSV file.

    _make_interpolator_on(on: str) -> RectBivariateSpline:
        Create a 2D interpolation function for a specific column ('on') in the table.

    make_interpolators() -> dict[str, RectBivariateSpline]:
        Create interpolation functions for all columns specified in 'to_interp'.

    generate_table() -> pl.DataFrame:
        Abstract method to generate the look-up table. To be implemented in subclasses.
    """

    def __init__(
        self,
        key1: str,
        key2: str,
        to_interp: list[str],
        cache_fn: Path,
        regenerate: bool = False,
        s=0.001,
    ):
        """
        Initialize CachedLUT with specified parameters.

        Parameters:
        -----------
        key1 : str
            First key used for table indexing.
        key2 : str
            Second key used for table indexing.
        to_interp : list[str]
            List of column names in the table to perform interpolation on.
        cache_fn : Path
            Path to the CSV file used for caching the table data.
        regenerate : bool, optional
            Flag indicating whether to regenerate the table even if cache file exists (default is False).
        s : float, optional
            Smoothing factor used in interpolation (default is 0.001).
        """
        self.key1 = key1
        self.key2 = key2
        self.to_interp = to_interp
        self.cache_fn = cache_fn
        self.s = s

        if cache_fn.exists() and not regenerate:
            self.df = self.load()
        else:
            self.df = self.generate_table()
            self.save()

        self.interpolators = self.make_interpolators()

    @abstractmethod
    def generate_table(self) -> pl.DataFrame:
        """
        Abstract method to generate the look-up table.

        This method should be implemented in subclasses to define how the table
        is generated based on specific requirements.

        Returns:
        --------
        pl.DataFrame:
            The generated look-up table.
        """
        pass

    def load(self) -> pl.DataFrame:
        """
        Load the cached table data from the CSV file.

        Returns:
        --------
        pl.DataFrame:
            The loaded DataFrame from the CSV.
        """
        return pl.read_csv(self.cache_fn)

    def save(self) -> None:
        """
        Save the current table data to the CSV file.

        Returns:
        --------
        None
        """
        self.df.write_csv(self.cache_fn)

    def _make_interpolator_on(self, on: str) -> RectBivariateSpline:
        """
        Create a 2D interpolation function for a specific column ('on') in the table.

        Parameters:
        -----------
        on : str
            Column name for which to create the interpolation function.

        Returns:
        --------
        RectBivariateSpline:
            Interpolation function for the specified column.
        """
        df_piv = (
            self.df.sort(self.key1, self.key2)
            .pivot(index=self.key2, columns=self.key1, values=on, maintain_order=True)
            .sort(self.key2)
        )

        x = np.array(df_piv.columns[1:], dtype=float).copy()
        y = df_piv[self.key2].to_numpy().copy()
        z = df_piv.to_numpy()[:, 1:].copy()

        interp = RectBivariateSpline(y, x, z, s=self.s)
        return interp

    def make_interpolators(self) -> dict[str, RectBivariateSpline]:
        """
        Create interpolation functions for all columns specified in 'to_interp'.

        Returns:
        --------
        dict[str, RectBivariateSpline]:
            Dictionary where keys are column names and values are interpolation functions.
        """
        interpolators = {}
        for key in self.to_interp:
            interpolators[key] = self._make_interpolator_on(key)
        return interpolators


def func_Ctprime(x) -> dict:
    Ctprime, yaw = x
    sol = model_Ctprime(Ctprime, np.deg2rad(yaw))
    return dict(
        yaw=np.round(yaw, 2),
        Ctprime=Ctprime,
        Cp=sol.Cp,
        Ct=sol.Ct,
        an=sol.an,
        u4=sol.u4,
        v4=sol.v4,
        x0=sol.x0,
        dp=sol.dp,
        dp_NL=sol.dp_NL,
    )


class UnifiedMomentumLUT(CachedLUT, MomentumBase):
    def __init__(self, cache_fn: Path = CACHE_FN_CTPRIME, regenerate=False, s=0.001):
        super().__init__(
            "yaw",
            "Ctprime",
            ["Cp", "Ct", "an", "u4", "v4", "x0", "dp"],
            cache_fn,
            regenerate=regenerate,
            s=s,
        )

    def generate_table(self) -> pl.DataFrame:
        Ctprimes = np.concatenate(
            [np.linspace(-1, 12, 200), np.linspace(12, 4000, 100), [50000, 100000]]
        )
        yaws = np.arange(-50.0, 50.1, 2.0)
        params = list(itertools.product(Ctprimes, yaws))

        # Run unified model and variations
        results = foreach(func_Ctprime, params, parallel=True)

        return pl.from_dicts(results).unique(["yaw", "Ctprime"])

    def __call__(self, Ctprime: ArrayLike, yaw: ArrayLike) -> MomentumSolution:
        yaw_deg = np.rad2deg(yaw)

        return MomentumSolution(
            Ctprime,
            yaw,
            self.interpolators["an"](Ctprime, yaw_deg, grid=False),
            self.interpolators["u4"](Ctprime, yaw_deg, grid=False),
            self.interpolators["v4"](Ctprime, yaw_deg, grid=False),
            self.interpolators["x0"](Ctprime, yaw_deg, grid=False),
            self.interpolators["dp"](Ctprime, yaw_deg, grid=False),
        )


def func_Ct(x) -> dict:
    Ct, yaw = x
    sol = model_Ct(Ct, np.deg2rad(np.round(yaw, 2)))
    return dict(
        yaw=np.round(yaw, 2),
        Ctprime=sol.Ctprime,
        Cp=sol.Cp,
        Ct=Ct,
        an=sol.an,
        u4=sol.u4,
        v4=sol.v4,
        x0=sol.x0,
        dp=sol.dp,
        dp_NL=sol.dp_NL,
    )


class ThrustBasedUnifiedMomentumLUT(CachedLUT, MomentumBase):
    def __init__(self, cache_fn: Path = CACHE_FN_CT, regenerate=False, s=0.025):
        super().__init__(
            "yaw",
            "Ct",
            ["Cp", "Ctprime", "an", "u4", "v4", "x0", "dp"],
            cache_fn,
            regenerate=regenerate,
            s=s,
        )

    def generate_table(self) -> pl.DataFrame:
        Cts = np.linspace(-1, 1.5, 100)
        yaws = np.arange(-50.0, 50.1, 2.0)
        params = list(itertools.product(Cts, yaws))

        # Run unified model and variations
        results = foreach(func_Ct, params, parallel=True)

        return pl.from_dicts(results).unique(["yaw", "Ct"])

    def __call__(self, Ct: ArrayLike, yaw: ArrayLike) -> MomentumSolution:
        yaw_deg = np.rad2deg(yaw)
        return MomentumSolution(
            self.interpolators["Ctprime"](Ct, yaw_deg, grid=False),
            yaw,
            self.interpolators["an"](Ct, yaw_deg, grid=False),
            self.interpolators["u4"](Ct, yaw_deg, grid=False),
            self.interpolators["v4"](Ct, yaw_deg, grid=False),
            self.interpolators["x0"](Ct, yaw_deg, grid=False),
            self.interpolators["dp"](Ct, yaw_deg, grid=False),
        )


class UnifiedLUTAD(Rotor):
    """
    Unified Momentum Model rotor

    Methods:
    - __call__(Ctprime, yaw): Calculate the rotor solution for given Ctprime and yaw inputs.
    """

    def __init__(self, rotor_grid: RotorGrid = None):
        """
        Initialize the UnifiedAD rotor model with the given axial induction factor.

        Parameters:
        - beta (float): Axial induction factor (default is 0.1403).
        """
        if rotor_grid is None:
            self.rotor_grid = Area()
        else:
            self.rotor_grid = rotor_grid
        self._model = UnifiedMomentumLUT()

    def __call__(
        self, x: float, y: float, z: float, windfield: Windfield, Ctprime, yaw
    ) -> RotorSolution:
        """
        Calculate the rotor solution for given Ctprime and yaw inputs.

        Parameters:
        - Ctprime (float): Thrust coefficient including the effect of yaw.
        - yaw (float): Yaw angle of the rotor.

        Returns:
        RotorSolution: The calculated rotor solution.
        """
        sol: MomentumSolution = self._model(Ctprime, yaw)

        # Get the points over rotor to be sampled in windfield
        xs_loc, ys_loc, zs_loc = self.rotor_grid.grid_points()
        xs_glob, ys_glob, zs_glob = xs_loc + x, ys_loc + y, zs_loc + z

        # sample windfield and calculate rotor effective wind speed
        Us = windfield.wsp(xs_glob, ys_glob, zs_glob)
        TIs = windfield.TI(xs_glob, ys_glob, zs_glob)

        REWS = self.rotor_grid.average(Us)
        RETI = np.sqrt(self.rotor_grid.average(TIs**2))

        # rotor solution is normalised by REWS. Convert normalisation to U_inf and return
        return RotorSolution(
            yaw,
            sol.Cp * REWS**3,
            sol.Ct * REWS**2,
            sol.Ctprime,
            sol.an * REWS,
            sol.u4 * REWS,
            sol.v4 * REWS,
            REWS,
            TI=RETI,
            extra=sol,
        )


class BEMUnifiedMomentumLUT(MITRotor.Momentum.MomentumModel):
    def __init__(self, averaging: Literal["sector", "annulus", "rotor"] = "rotor"):
        if averaging == "rotor":
            self._func = self._func_rotor
        elif averaging == "annulus":
            self._func = self._func_annulus
        elif averaging == "sector":
            self._func = self._func_sector
        else:
            raise ValueError(
                f"Averaging method {averaging} not found for BEMUnifiedMomentumLUT model."
            )
        self.averaging = averaging
        self.model = ThrustBasedUnifiedMomentumLUT()

    def Ct_a(self, Ct: ArrayLike, yaw: float) -> ArrayLike:

        sol = self.model(Ct, yaw)
        return sol.an

    def __call__(
        self,
        aero_props: "MITRotor.AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "MITRotor.RotorDefinition",
        geom: "MITRotor.BEMGeometry",
    ) -> ArrayLike:
        an = self._func(aero_props, pitch, tsr, yaw, rotor, geom)
        return an

    def _func_rotor(
        self,
        aero_props: "MITRotor.AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "MITRotor.RotorDefinition",
        geom: "MITRotor.BEMGeometry",
    ) -> ArrayLike:
        Ct = aero_props.solidity * aero_props.W**2 * aero_props.Cax
        Ct_rotor = geom.rotor_average(geom.annulus_average(Ct))

        a_target = self.Ct_a(Ct_rotor, yaw)

        a_new = aero_props.F
        a_rotor = geom.rotor_average(geom.annulus_average(a_new))
        a_new *= a_target / a_rotor

        return a_new

    def _func_annulus(
        self,
        aero_props: "MITRotor.AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "MITRotor.RotorDefinition",
        geom: "MITRotor.BEMGeometry",
    ) -> ArrayLike:
        Ct = geom.annulus_average(
            aero_props.solidity * aero_props.W**2 * aero_props.Cax
        )
        _Ct = np.clip(Ct, -1, 1.59)
        a = self.Ct_a(_Ct, yaw)[:, None] * np.ones(geom.shape)

        return a

    def _func_sector(
        self,
        aero_props: "MITRotor.AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "MITRotor.RotorDefinition",
        geom: "MITRotor.BEMGeometry",
    ) -> ArrayLike:
        Ct = aero_props.solidity * aero_props.W**2 * aero_props.Cax
        ans = self.Ct_a(Ct.ravel(), yaw)
        return ans.reshape(geom.shape)


# simple Cosine model rotor

class CosineAD(Rotor):
    """
    Simple Cosine model rotor. Uses Shapiro lifting line model for v4. 

    Methods:
    - __call__(Ctprime, yaw): Calculate the rotor solution for given Ctprime and yaw inputs.
    """

    def __init__(self, rotor_grid: RotorGrid = None, Pp: float = 3.):
        """
        Initialize the UnifiedAD rotor model with the given axial induction factor.

        Parameters:
        - beta (float): Axial induction factor (default is 0.1403).
        """
        if rotor_grid is None:
            self.rotor_grid = Area()
        else:
            self.rotor_grid = rotor_grid
        self._model = LimitedHeck()
        self.Pp = Pp

    def __call__(
        self, x: float, y: float, z: float, windfield: Windfield, Ctprime, yaw
    ) -> RotorSolution:
        """
        Calculate the rotor solution for given Ctprime and yaw inputs.

        Parameters:
        - Ctprime (float): Thrust coefficient including the effect of yaw.
        - yaw (float): Yaw angle of the rotor.

        Returns:
        RotorSolution: The calculated rotor solution.
        """
        sol: MomentumSolution = self._model(Ctprime, yaw)
        sol_1d = self._model(Ctprime, yaw * 0)  # 1d solution

        # Get the points over rotor to be sampled in windfield
        xs_loc, ys_loc, zs_loc = self.rotor_grid.grid_points()
        xs_glob, ys_glob, zs_glob = xs_loc + x, ys_loc + y, zs_loc + z

        # sample windfield and calculate rotor effective wind speed
        Us = windfield.wsp(xs_glob, ys_glob, zs_glob)
        TIs = windfield.TI(xs_glob, ys_glob, zs_glob)

        REWS = self.rotor_grid.average(Us)
        RETI = np.sqrt(self.rotor_grid.average(TIs**2))

        # rotor solution is normalised by REWS. Convert normalisation to U_inf and return
        return RotorSolution(
            yaw,
            sol_1d.Cp * np.cos(sol.yaw)**self.Pp * REWS**3,
            sol.Ct * REWS**2,
            sol.Ctprime,
            sol.an * REWS,
            sol.u4 * REWS,
            sol.v4 * REWS,
            REWS,
            TI=RETI,
            extra=sol,
        )

# class AD(Rotor):
#     """
#     Axial Distribution rotor model.

#     Methods:
#     - __call__(Ctprime, yaw): Calculate the rotor solution for given Ctprime and yaw inputs.
#     """

#     def __init__(self, rotor_grid: RotorGrid = None):
#         """
#         Initialize the AD rotor model using the Heck momentum model.
#         """
#         self._model = Heck()
#         if rotor_grid is None:
#             self.rotor_grid = Area()
#         else:
#             self.rotor_grid = rotor_grid

#     def __call__(self, x: float, y: float, z: float, windfield: Windfield, Ctprime, yaw) -> RotorSolution:
#         """
#         Calculate the rotor solution for given Ctprime and yaw inputs.

#         Parameters:
#         - Ctprime (float): Thrust coefficient including the effect of yaw.
#         - yaw (float): Yaw angle of the rotor.

#         Returns:
#         RotorSolution: The calculated rotor solution.
#         """
#         # Calculate rotor solution (independent of wind field in this model)
#         sol: MomentumSolution = self._model(Ctprime, yaw)

#         # Get the points over rotor to be sampled in windfield
#         xs_loc, ys_loc, zs_loc = self.rotor_grid.grid_points()
#         xs_glob, ys_glob, zs_glob = xs_loc + x, ys_loc + y, zs_loc + z

#         # sample windfield and calculate rotor effective wind speed
#         Us = windfield.wsp(xs_glob, ys_glob, zs_glob)
#         TIs = windfield.TI(xs_glob, ys_glob, zs_glob)
        
#         REWS = self.rotor_grid.average(Us)
#         RETI = np.sqrt(self.rotor_grid.average(TIs**2))

#         # rotor solution is normalised by REWS. Convert normalisation to U_inf and return
#         return RotorSolution(
#             yaw,
#             sol.Cp * REWS**3,
#             sol.Ct * REWS**2,
#             sol.Ctprime,
#             sol.an * REWS,
#             sol.u4 * REWS,
#             sol.v4 * REWS,
#             REWS,
#             TI=RETI,
#             extra=sol,
#         )

