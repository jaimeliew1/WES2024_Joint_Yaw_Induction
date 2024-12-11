from dualitic import DualVariables

from pathlib import Path
import polars as pl
from WES2024 import utils
from WES2024.CustomRotors import UnifiedLUTAD, CosineAD
from WES2024.LES.run import TIAMB, SimulationDefinition
from WES2024.optimise import (
    JointControl,
    NoControl,
    ThrustControl,
    YawControl,
)
from rich import print
from mitwindfarm import (
    Windfarm,
    Square,
)

# from WES2024.LES.step_2_calibrate import VariableKwGaussianWakeModel2
from WES2024.LES.run_old import VariableKwGaussianWakeModel
from mitwindfarm.Wake import GaussianWakeModel
from mitwindfarm.Rotor import AD

LAYOUT = Square(6.0, 5).rotate(45).rotate(-2.5)

# Controller optimisers
CONTROLLERS = {
    "nocontrol": NoControl,
    "yawcontrol": YawControl,
    "thrustcontrol": ThrustControl,
    "jointcontrol": JointControl,
}

ROTOR_MODEL = {
    "cosine": CosineAD(),
    "cosineTI": CosineAD(),
    "jfm": AD(),
    "jfmTI": AD(),
    "unified": UnifiedLUTAD(),
    "unifiedTI": UnifiedLUTAD(),
}

VarGauss = VariableKwGaussianWakeModel(0, 0.90682, -0.00152)
UniGauss =VariableKwGaussianWakeModel(0, 0.91948314, -0.00896332)  # unified setpoints from calibration
WAKE_MODEL = {
    "cosine": GaussianWakeModel(),
    "cosineTI": VarGauss,
    "jfm": GaussianWakeModel(),
    "jfmTI": VarGauss,
    "unified": UniGauss,
    "unifiedTI": UniGauss,
}

CASE_NOTE = "Adding cosine model"


def run(rotor_key: str, output_path: Path) -> None:

    # Initialise MITWindfarm using VariableKwGaussianWakeModel
    wakemodel = GaussianWakeModel()  # update to normal wake and cosine rotor models
    windfarm = Windfarm(
        rotor_model=ROTOR_MODEL[rotor_key], 
        wake_model=WAKE_MODEL[rotor_key], 
        TIamb=TIAMB
    )

    # For each controller to optimise...
    for name, controller in CONTROLLERS.items():
        filestem = f"diamond_wdir-2.5_{rotor_key}_{name}"

        # Run optimisation
        sol = controller(LAYOUT, windfarm).optimise(use_gradients=True, verbose=True)

        # write simulation definition for LES based on optimal setpoints
        definition = SimulationDefinition.from_windfarmsolution(
            sol,
            casename=filestem,
            wdir=-2.5,
            controller=name,
            case_note=CASE_NOTE,
        )
        definition.write_json(output_path / f"{filestem}.json")

        # write MITWindfarm results to csv file
        _df = utils.to_polars(sol).with_columns(
            pl.lit(name).alias("controller"),
            pl.lit(-2.5).alias("wdir"),
            pl.lit("MITWindfarm").alias("simulator"),
        )
        _df.write_csv(output_path / f"{filestem}.csv")


if __name__ == "__main__":
    path = Path(__file__).parent / "LES_newinput"
    path.mkdir(exist_ok=True, parents=True)

    # # All of Jaime's cases have the TI calibration... comment these out: 
    # run("cosine", path)
    # run("jfm", path)
    # run("unified", path)

    # cases which include TI calibration: 
    run("cosineTI", path)
    run("jfmTI", path)
    run("unifiedTI", path)
