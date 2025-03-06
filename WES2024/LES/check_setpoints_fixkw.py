from dualitic import DualVariables

from pathlib import Path
import polars as pl
import numpy as np
from WES2024 import utils
from WES2024.CustomRotors import UnifiedLUTAD
from WES2024.LES.run import TIAMB, SimulationDefinition
from WES2024.optimise import (
    JointControl,
    NoControl,
    ThrustControl,
    YawControl,
)
from rich import print
from mitwindfarm import (
    AD, 
    GaussianWakeModel, 
    Windfarm,
    Square,
)

LAYOUT = Square(6.0, 5).rotate(45).rotate(-2.5)

# Controller optimisers
CONTROLLERS = {
    "nocontrol": NoControl,
    "yawcontrol": YawControl,
    "thrustcontrol": ThrustControl,
    "jointcontrol": JointControl,
}

CASE_NOTE = "Test recreating thrust control solution"


def run(calibration_key: str, output_path: Path) -> None:
    """Compute optimal set points and write to disk"""
    wakemodel = GaussianWakeModel(sigma=1/np.sqrt(8))  # default params
    windfarm = Windfarm(
        rotor_model=AD(), 
        wake_model=wakemodel, 
        TIamb=TIAMB
    )

    # For each controller to optimise...
    for name, controller in CONTROLLERS.items():
        filestem = f"wdir-2.5_{calibration_key}_{name}"

        # Run optimisation
        sol = controller(LAYOUT, windfarm).optimise(use_gradients=True, verbose=True)

        # write simulation definition for LES based on optimal setpoints
        definition = SimulationDefinition.from_windfarmsolution(
            sol,
            casename=f"diamond_wdir-2.5_{name}",
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
    # run("ManualCal", STEP_3_DIR)
    # run("AutoCal", STEP_3_DIR)
    # pass
    path = Path(__file__).parent / "debug"
    path.mkdir(exist_ok=True, parents=True)

    run("fixedkw", path)
