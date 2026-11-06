"""
Wake-model factory for the R1 revision, adapted from tandem_model
(models.py / utils.py). Solver keys:

- "gauss":     original WES2024 calibrated axisymmetric Gaussian (Shapiro et al. 2018)
- "gaussBP":   Bastankhah and Porte-Agel (2016) non-axisymmetric Gaussian
- "varvortex": vortex-sheet model with veer and curling effects
- "tandem":    TANDEM numerically integrated (curled) wake model
"""

import mitwindfarm as mitwf
import polars as pl
from MITRotor import DefaultTangentialInduction, IEA15MW, IEA3_4MW, NREL5MW
from MITRotor.Aerodynamics import DefaultAerodynamics

from WES2024.BEM_gradients import DualBEM
from WES2024.CustomRotors import (
    BEMCosine,
    BEMLUT,
    BEMLUTCosine,
    BEMTI,
    BEMUnifiedMomentumLUT,
    BEMUnifiedMomentumLUTAnnulus,
    UnifiedLUTAD,
)
from WES2024.LES.shared import TIAMB
from WES2024.LES_new.final_calibration import get_wakemodel as get_calibrated_gauss

DISPLAY_NAMES = {
    "gauss": "Axisymmetric Gaussian",
    "gaussBP": "Gaussian",
    "varvortex": "Vortex sheet",
    "tandem": "TANDEM",
    "scott": "Scott",
}

# marching settings for the parabolized (curled) RANS closures
CURLED_MODEL_KWARGS = dict(
    integrator="scipy_rk23",
    sigma_0_fw=0.21,
    auto_expand=True,
    ybuff=2,
    dy=0.1,
    dz=0.1,
    use_r4=None,
)
# nw_blend > 0 smooths the hard near/far-wake switch at nw_mask = 0.95 (window centred on 0.95). With a hard
# switch Cp is a sawtooth in the control variables and AD misses the moving-boundary term (see debug_sawtooth.py).
K_KWARGS = {"tandem": dict(C_nu=0.35, l_eps=0.78, C_w=4, nw_blend=0.01)}


def get_bem_rotor(geometry=None, rotor_class=BEMTI, **rotor_kwargs):
    """IEA 15 MW BEM rotor with the unified momentum LUT; same construction as Generate/two_turbine_BEM.py.

    `geometry` is an optional MITRotor BEMGeometry(Nr, Ntheta); the default is DualBEM's own (Nr=128, Ntheta=12).
    `rotor_class` (BEMTI or a subclass, e.g. BEMCosine) and `rotor_kwargs` (e.g. Pp) select the power model.
    """
    return rotor_class(
        IEA15MW(),
        BEM_model=DualBEM,
        **(dict(geometry=geometry) if geometry is not None else {}),
        momentum_model=BEMUnifiedMomentumLUT(averaging="rotor_induction_tiploss"),
        aerodynamic_model=DefaultAerodynamics(),
        tangential_induction_model=DefaultTangentialInduction(),  # mu^1 and Prandtl tip/root loss F in a'
        **rotor_kwargs,
    )


def get_bem_cosine_rotor(Pp=1.88, geometry=None):
    """BEM rotor with the cosine laws Cp = Cp0 cos^Pp(yaw), Ct = Ct0 cos^Tp(yaw), Tp = 2 Pp / 3 (CustomRotors.BEMCosine)."""
    return get_bem_rotor(geometry, rotor_class=BEMCosine, Pp=Pp)


def get_bem_annulus_rotor(geometry=None, rotor_class=BEMTI, **rotor_kwargs):
    """
    Like get_bem_rotor, but closes the induction with the standard local (per-annulus) tip-loss correction
    Cx_corr = Cx/F -- clipped to Ct <= 1.5 where F -> 0 at root/tip -- through the Unified Momentum Model's LUT,
    instead of the disk-integrated-then-redistributed "rotor_induction_tiploss" averaging get_bem_rotor uses. This
    is the architecture classical BEM codes use (AeroDyn, CCBlade: Prandtl F + an empirical high-thrust correction),
    with UMM standing in for that correction (see CustomRotors.BEMUnifiedMomentumLUTAnnulus).
    """
    return rotor_class(
        IEA15MW(),
        BEM_model=DualBEM,
        **(dict(geometry=geometry) if geometry is not None else {}),
        momentum_model=BEMUnifiedMomentumLUTAnnulus(),
        aerodynamic_model=DefaultAerodynamics(),
        tangential_induction_model=DefaultTangentialInduction(),
        **rotor_kwargs,
    )


def get_bem_cosine_annulus_rotor(Pp=1.88, geometry=None):
    """Annulus-closure counterpart of get_bem_cosine_rotor (CustomRotors.BEMCosine with the annulus-averaged
    tip-loss closure of get_bem_annulus_rotor, instead of get_bem_rotor's "rotor_induction_tiploss" closure)."""
    return get_bem_annulus_rotor(geometry, rotor_class=BEMCosine, Pp=Pp)


def get_bem_lut_rotor(lut_path=None, rotor_grid=None, rotor_class=BEMLUT, **rotor_kwargs):
    """
    Tabulated uniform-inflow BEM rotor (BEMLUT), ~25x cheaper than get_bem_rotor in AD optimisations of a 25-turbine
    farm and independent of the BEM resolution. The table comes from R1/generate/generate_bem_lut_annulus.py (generated on
    first use; annulus-averaged UMM-LUT closure, get_bem_annulus_rotor). The original table (get_bem_rotor's
    "rotor_induction_tiploss" closure, tmp/extra/generate_bem_lut.py, pitch +-10 deg / TSR 3-10 / yaw +-45 deg) is left
    untouched on disk and still loads fine through this same function by passing its lut_path explicitly.
    """
    from WES2024.R1.generate.generate_bem_lut_annulus import generate, lut_path as default_path

    if lut_path is None:
        generate()  # no-op when cached
        lut_path = default_path()
    return rotor_class(pl.read_csv(lut_path), rotor_grid=rotor_grid, **rotor_kwargs)


def get_bem_lut_cosine_rotor(Pp=1.88, lut_path=None, rotor_grid=None):
    """Tabulated counterpart of get_bem_cosine_rotor (CustomRotors.BEMLUTCosine)."""
    return get_bem_lut_rotor(lut_path, rotor_grid, rotor_class=BEMLUTCosine, Pp=Pp)


def get_windfarm(key: str, TIamb: float = TIAMB, rotor_model=None, solver_overrides: dict | None = None, inflow=None):
    """
    Returns a mitwindfarm Windfarm (or CurledWindfarm) for solver key `key`. `solver_overrides` updates the solver
    kwargs of the marching models (e.g. dict(dy=0.2, dz=0.2) for a half-resolution cross-stream grid). TANDEM includes
    the ground (bottom_wall_z = -1, one D below the hub at z = 0) unless overridden with dict(bottom_wall_z=None).
    `inflow`: optional mitwf Windfield (e.g. R1.get_les_inflow_windfarm.get_les_inflow_windfarm(), the LES precursor
    profiles) used as the ambient inflow instead of Uniform(TIamb). It carries its own TI profile, so `TIamb` is then
    ignored (the wake models read TI from the windfield). Supported for "gaussBP", "varvortex" and "tandem"; the
    default (None) is unchanged.
    """
    rotor_model = UnifiedLUTAD() if rotor_model is None else rotor_model
    custom_inflow = inflow is not None
    if custom_inflow and key not in ("gaussBP", "varvortex", "tandem"):
        raise ValueError(f"inflow is not supported for wake model {key!r}")
    inflow = inflow if custom_inflow else mitwf.Uniform(TIamb=TIamb)
    TIamb = None if custom_inflow else TIamb  # Windfarm / wakes take TI from the windfield

    if key == "gauss":
        return mitwf.Windfarm(
            rotor_model=rotor_model,
            wake_model=get_calibrated_gauss(),
            TIamb=TIamb,
            superposition=mitwf.Niayifar(),
        )
    if key == "gaussBP":
        wake_model = mitwf.VariableKwGaussBPWakeModel(windfield=inflow)
    elif key == "varvortex":
        wake_model = mitwf.VariableVortexWakeModel(windfield=inflow, z_wall=-1, ustar=0.0312)
    elif key == "tandem":
        solver_kwargs = dict(
            k_model="tandem",
            k_kwargs=K_KWARGS["tandem"],
            u_model="upwind",
            verbose=False,
            bottom_wall_z=-1,
            **CURLED_MODEL_KWARGS,
        )
        solver_kwargs.update(solver_overrides or {})
        return mitwf.CurledWindfarm(
            rotor_model=rotor_model, base_windfield=inflow, solver_kwargs=solver_kwargs
        )
    elif key == "scott":
        solver_kwargs = dict(
            k_model="scott",
            u_model="default",
            **CURLED_MODEL_KWARGS,
        )
        solver_kwargs.update(solver_overrides or {})
        return mitwf.CurledWindfarm(
            rotor_model=rotor_model, base_windfield=inflow, solver_kwargs=solver_kwargs
        )
    else:
        raise ValueError(f"Unknown wake model: {key}")

    return mitwf.Windfarm(
        rotor_model=rotor_model,
        wake_model=wake_model,
        base_windfield=inflow,
        superposition=mitwf.Niayifar(),
        TIamb=TIamb,
    )
