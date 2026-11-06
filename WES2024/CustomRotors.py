from abc import ABC, abstractmethod
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal
import itertools
import warnings
from pathlib import Path
from typing import Literal

import MITRotor
import numpy as np

# MITRotor.Geometry calls np.trapezoid directly (added in numpy 2.0); this venv's
# numpy 1.26.4 only has the deprecated np.trapz alias it replaced.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

import polars as pl
from foreach import foreach
from mitwindfarm import Area, Line, Point, RotorSolution
from mitwindfarm.Rotor import BEM, Rotor, compute_x0_with_TI
from mitwindfarm.RotorGrid import RotorGrid
from mitwindfarm.Windfield import Windfield
from numpy.typing import ArrayLike
from dualitic import DualNumber
from scipy.interpolate import NdBSpline, RectBivariateSpline, make_interp_spline
from UnifiedMomentumModel.Momentum import (
    MomentumBase,
    MomentumSolution,
    UnifiedMomentum,
    ThrustBasedUnified,
    LimitedHeck,
)

model_Ctprime = UnifiedMomentum()
model_Ct = ThrustBasedUnified()

if TYPE_CHECKING:
    from MITRotor.Geometry import BEMGeometry
    from MITRotor.RotorDefinition import RotorDefinition
    from MITRotor.Aerodynamics import AerodynamicProperties

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
        rotor_sol = RotorSolution(
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
        # The LUT is built without turbulence, so its x0 is the TI-free near-wake length. Replace it with
        # the TI-dependent value (consistent with the unified momentum model coupled with TI).
        sol.x0 = compute_x0_with_TI(rotor_sol)
        return rotor_sol


class BEMTI(BEM):
    """
    BEM rotor whose solution carries the TI-dependent near-wake length x0, computed with compute_x0_with_TI
    from the rotor-averaged induction and initial wake velocity (DualBEM provides only the TI-free LUT x0).
    """

    def __call__(self, *args, **kwargs) -> RotorSolution:
        rotor_sol = super().__call__(*args, **kwargs)
        bem_sol = rotor_sol.extra
        # compute_x0_with_TI reads the non-dimensional induction and u4 from `extra`, but BEMSolution.a() is a method
        proxy = SimpleNamespace(
            yaw=rotor_sol.yaw,
            tilt=rotor_sol.tilt,
            TI=rotor_sol.TI,
            extra=SimpleNamespace(an=bem_sol.a(), u4=bem_sol.u4),
        )
        bem_sol.x0 = compute_x0_with_TI(proxy)
        return rotor_sol


class BEMLUTTable:
    """
    Tensor-product cubic-spline interpolation of the uniform-inflow BEM table (R1/generate_bem_lut.py):
    (pitch [rad], TSR, yaw [rad]) -> Cp, Ct, Ctprime, an, u4, v4, all normalised by the rotor-effective wind speed.
    Dual-number inputs are propagated with the analytic spline derivatives. Inputs outside the table domain (the
    bounds of the BEM controllers) are clipped.
    """

    OUTPUTS = ("Cp", "Ct", "Ctprime", "an", "u4", "v4")

    def __init__(self, df: pl.DataFrame, show_warnings: bool = True):
        df = df.sort("pitch", "tsr", "yaw")
        n_bad = (~df["converged"]).sum()
        if n_bad and show_warnings:
            bad = df.filter(~pl.col("converged"))
            warnings.warn(
                f"BEM table has {n_bad} non-converged points (pitch {bad['pitch'].min():g} to "
                f"{bad['pitch'].max():g} deg, tsr {bad['tsr'].min():g} to {bad['tsr'].max():g}); "
                "using the tabulated values as-is (they are smooth with their converged neighbours)."
            )
        axes = [np.unique(df["pitch"].to_numpy()), np.unique(df["tsr"].to_numpy()), np.unique(df["yaw"].to_numpy())]
        self.axes = [np.deg2rad(axes[0]), axes[1], np.deg2rad(axes[2])]
        values = np.stack([df[k].to_numpy() for k in self.OUTPUTS], axis=-1).reshape(*(len(a) for a in axes), -1)
        c, knots = values, []
        for a, x in enumerate(self.axes):  # exact tensor-product interpolation, one axis at a time
            bs = make_interp_spline(x, c, k=3, axis=a)
            c = np.moveaxis(bs.c, 0, a)
            knots.append(bs.t)
        self._spline = NdBSpline(tuple(knots), c, k=3)
        self.lo = np.array([a[0] for a in self.axes])
        self.hi = np.array([a[-1] for a in self.axes])

    @staticmethod
    def _split(v):
        if isinstance(v, DualNumber):
            return float(v.real.ravel()[0]), v.dual.reshape(-1, v.dual.shape[-1])[0]
        return float(v), None

    def __call__(self, pitch, tsr, yaw) -> dict:
        parts = [self._split(v) for v in (pitch, tsr, yaw)]
        x = np.clip([p[0] for p in parts], self.lo, self.hi)[None, :]
        values = self._spline(x)[0]
        duals = [p[1] for p in parts]
        n = next((len(d) for d in duals if d is not None), None)
        if n is None:
            return dict(zip(self.OUTPUTS, values))
        jac = np.stack([self._spline(x, nu=tuple(int(i == j) for i in range(3)))[0] for j in range(3)], axis=-1)  # (out, 3)
        out = {}
        for k, name in enumerate(self.OUTPUTS):
            grad = sum(jac[k, j] * d for j, d in enumerate(duals) if d is not None)
            out[name] = DualNumber(values[k], grad)
        return out


class BEMLUT(Rotor):
    """
    BEM rotor in uniform inflow, tabulated at a converged radial resolution (see BEMLUTTable). Like UnifiedLUTAD, the
    rotor-effective wind speed is the average of the windfield over a small rotor grid and the table supplies the
    REWS-normalised rotor response, so the cost per rotor is independent of the BEM resolution. Neglects shear, veer and
    wake deficits across the disk (the BEM rotor resolves them); set points are (pitch, TSR, yaw) as for BEM.
    """

    def __init__(self, table: BEMLUTTable | pl.DataFrame, rotor_grid: RotorGrid = None, show_warnings: bool = True):
        """`show_warnings=False` silences the "non-converged points" warning of BEMLUTTable (when `table` is a DataFrame)."""
        self.table = table if isinstance(table, BEMLUTTable) else BEMLUTTable(table, show_warnings=show_warnings)
        self.rotor_grid = Area() if rotor_grid is None else rotor_grid

    def __call__(self, x: float, y: float, z: float, windfield: Windfield, pitch, tsr, yaw=0, tilt=0) -> RotorSolution:
        if np.any(np.asarray(getattr(tilt, "real", tilt)) != 0):
            raise NotImplementedError("BEMLUT does not support tilt.")
        t = self.table(pitch, tsr, yaw)
        xs_loc, ys_loc, zs_loc = self.rotor_grid.grid_points()
        xs, ys, zs = xs_loc + x, ys_loc + y, zs_loc + z
        Us = windfield.wsp(xs, ys, zs)
        TIs = windfield.TI(xs, ys, zs)
        REWS = self.rotor_grid.average(Us)
        RETI = np.sqrt(self.rotor_grid.average(TIs**2))
        sol = SimpleNamespace(pitch=pitch, tsr=tsr, yaw=yaw, **t)
        rotor_sol = RotorSolution(
            yaw,
            t["Cp"] * REWS**3,
            t["Ct"] * REWS**2,
            t["Ctprime"],
            t["an"] * REWS,
            t["u4"] * REWS,
            t["v4"] * REWS,
            REWS,
            tilt=0.0,
            w4=0.0 * t["v4"] * REWS,
            TI=RETI,
            extra=sol,
        )
        # TI-dependent near-wake length, as for UnifiedLUTAD and BEMTI
        sol.x0 = compute_x0_with_TI(rotor_sol)
        return rotor_sol


class BEMCosine(BEMTI):
    """
    BEM rotor with the cosine yaw laws of Tamaro et al. (WES 2024): the yaw-aligned BEM solution at the same pitch, TSR and
    inflow is scaled with the yaw misalignment,

        Cp(pitch, TSR, yaw) = Cp(pitch, TSR, 0) cos(yaw)^Pp,    Ct(pitch, TSR, yaw) = Ct(pitch, TSR, 0) cos(yaw)^Tp,

    Tp = 2/3 Pp (Pp = 1.88 -> Tp = 1.253). The returned rotor solution carries the actual yaw, so the wake model deflects
    and weights the wake with the scaled Ct and the yaw. Ct', an, u4, v4 are those of the yaw-aligned solution (the
    Gaussian BP2016 wake only reads Ct and yaw; kw does not depend on Ct' with the default b = 0). No K-omega^2 law is
    involved: pitch and TSR are free set points.
    """

    def __init__(self, *args, Pp: float = 1.88, Tp: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.Pp = Pp
        self.Tp = Pp * 2 / 3 if Tp is None else Tp

    def __call__(self, x, y, z, windfield, pitch, tsr, yaw=0, tilt=0) -> RotorSolution:
        aligned = super().__call__(x, y, z, windfield, pitch, tsr, yaw * 0, tilt)
        c = np.cos(yaw)
        return replace(aligned, yaw=yaw, Cp=aligned.Cp * c**self.Pp, Ct=aligned.Ct * c**self.Tp)


class BEMLUTCosine(BEMLUT):
    """BEMCosine for the tabulated BEM rotor: the yaw-aligned values are read from the table at yaw = 0."""

    def __init__(self, *args, Pp: float = 1.88, Tp: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.Pp = Pp
        self.Tp = Pp * 2 / 3 if Tp is None else Tp

    def __call__(self, x, y, z, windfield, pitch, tsr, yaw=0, tilt=0) -> RotorSolution:
        aligned = super().__call__(x, y, z, windfield, pitch, tsr, yaw * 0, tilt)
        c = np.cos(yaw)
        return replace(aligned, yaw=yaw, Cp=aligned.Cp * c**self.Pp, Ct=aligned.Ct * c**self.Tp)


class BEMUnifiedMomentumLUT(MITRotor.Momentum.MomentumModel):
    def __init__(
        self,
        averaging: Literal[
            "sector", "annulus", "rotor", "rotor_induction_tiploss"
        ] = "rotor_induction_tiploss",
    ):
        if averaging == "rotor":
            self._func = self._func_rotor
        if averaging == "rotor_induction_tiploss":
            self._func = self._func_rotor_induction_tiploss
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

    def compute_induction(self, Ct: ArrayLike, yaw: float, tilt: float = 0.0) -> ArrayLike:
        self._check_no_tilt(tilt)
        sol = self.model(Ct, yaw)
        return sol.an

    def compute_initial_wake_velocities(
        self, Ct: ArrayLike, yaw: float, tilt: float = 0.0
    ) -> ArrayLike:
        self._check_no_tilt(tilt)
        sol = self.model(Ct, yaw)
        return sol.u4, sol.v4, 0.0 * sol.v4  # no vertical wake deflection without tilt

    def compute_x0(self, Ct: ArrayLike, yaw: float) -> ArrayLike:
        """Near-wake length from the same unified-momentum LUT that provides u4 and v4."""
        return self.model(Ct, yaw).x0

    @staticmethod
    def _check_no_tilt(tilt):
        if np.any(tilt != 0):
            raise NotImplementedError("BEMUnifiedMomentumLUT does not support tilt.")

    def __call__(
        self,
        aero_props: "MITRotor.AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "MITRotor.RotorDefinition",
        geom: "MITRotor.BEMGeometry",
        tilt: float = 0.0,
    ) -> ArrayLike:
        self._check_no_tilt(tilt)
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
        Ct = aero_props.solidity * aero_props.W**2 * aero_props.C_n

        Ct_rotor = geom.rotor_average(geom.annulus_average(Ct))

        a = self.Ct_a(Ct_rotor, yaw)

        return a

    def _func_rotor_induction_tiploss(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        Ct = aero_props.solidity * aero_props.W**2 * aero_props.C_n
        Ct_rotor = geom.rotor_average(geom.annulus_average(Ct))

        a_target = self.compute_induction(Ct_rotor, yaw)

        # Not in place: `a_new *= ...` on the float array would also rescale aero_props.F, which the tangential induction
        # model (DefaultTangentialInduction: C_tau / F) reads afterwards. Dual numbers rebind on `*=`, so float and
        # dual evaluations disagreed by ~1.6 % in Cp and AD != FD.
        a_rotor = geom.rotor_average(geom.annulus_average(aero_props.F))
        a_new = aero_props.F * (a_target / a_rotor)

        return a_new


class BEMUnifiedMomentumLUTAnnulus(BEMUnifiedMomentumLUT):
    """
    Annulus-averaged BEM momentum closure: the standard local tip-loss correction (Cx_corr = Cx/F,
    inverted independently at each annulus from its own local thrust) through the Unified Momentum
    Model (via its LUT) instead of Buhl/Heck's empirical high-thrust correction -- i.e. the same
    architecture classical BEM codes use (AeroDyn's BEMT module, CCBlade: Prandtl F + Buhl), just with
    UMM standing in for Buhl/Glauert. This matches the coupling described in Liew, Heck & Howland 2024
    ("classical momentum theory is directly replaced... there is no need to modify the blade element
    portion of BEM"), unlike BEMUnifiedMomentumLUT's default "rotor_induction_tiploss" averaging (one
    disk-integrated UMM solve, redistributed radially in proportion to F -- likely a numerical
    workaround for the issue below, not the published method).

    Cx_corr is clipped before being passed to the UMM: F -> 0 at the root and tip makes Cx_corr = Cx/F
    blow up without bound (e.g. Ct ~ 15 at the hub-center node for IEA15MW), which is both unphysical
    and outside the LUT's tabulated domain. Default Ct_clip matches ThrustBasedUnifiedMomentumLUT's own
    tabulated domain (Ct in [-1, 1.5], yaw in [-50, 50] deg).
    """

    # Single-turbine (yaw=0) unyawed optimum for *this* closure, computed 2026-10-26 via
    # R1/wakemodels.get_bem_annulus_rotor (Nr=128, Ntheta=12, not the LUT, to avoid baking in interpolation error):
    # JointControlBEM (SLSQP, AD) and a tight 5-start Nelder-Mead polish agree to ~6 digits (no flat-ridge ambiguity
    # here, unlike the old closure's history in optimise.py). Kept on the class, not in optimise.py, because
    # PITCH_OPT/TSR_OPT/CP_OPT there were derived for get_bem_rotor's "rotor_induction_tiploss" closure and are
    # specific to it; scripts using this closure should read these instead of those globals.
    #
    # Notably Cp is ~7.3% lower than the old closure's CP_OPT (0.535510 at pitch -1.0 deg, tsr 9.165275): the old
    # "rotor_induction_tiploss" averaging (one disk-integrated UMM solve redistributed radially by F, never divides
    # by F) appears to systematically over-predict Cp relative to the per-annulus local closure used here, which is
    # the architecture classical BEM codes (AeroDyn, CCBlade) actually use. Worth a closer look if it recurs elsewhere.
    PITCH_OPT = -0.005035990848671321  # rad (-0.288541 deg)
    TSR_OPT = 9.232356732737388
    CP_OPT = 0.49650303785016325

    def __init__(self, Ct_clip: tuple[float, float] = (-1.0, 1.5)):
        super().__init__(averaging="annulus")
        self.Ct_clip = Ct_clip

    def compute_induction(self, Ct: ArrayLike, yaw: float, tilt: float = 0.0) -> ArrayLike:
        return super().compute_induction(np.clip(Ct, *self.Ct_clip), yaw, tilt=tilt)

    def compute_initial_wake_velocities(self, Ct: ArrayLike, yaw: float, tilt: float = 0.0) -> ArrayLike:
        return super().compute_initial_wake_velocities(np.clip(Ct, *self.Ct_clip), yaw, tilt=tilt)

    def compute_x0(self, Ct: ArrayLike, yaw: float) -> ArrayLike:
        return super().compute_x0(np.clip(Ct, *self.Ct_clip), yaw)


# simple Cosine model rotor


class CosineAD(Rotor):
    """
    Simple Cosine model rotor. Uses Shapiro lifting line model for v4.

    Methods:
    - __call__(Ctprime, yaw): Calculate the rotor solution for given Ctprime and yaw inputs.
    """

    def __init__(self, rotor_grid: RotorGrid = None, Pp: float = 3.0):
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
            sol_1d.Cp * np.cos(sol.yaw) ** self.Pp * REWS**3,
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
