from dualitic import DualVariables

from pathlib import Path
import polars as pl
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
    VariableKwGaussianWakeModel,
    Windfarm,
    Square,
    Niayifar,
)

from WES2024.LES.shared import STEP_2_DIR, STEP_3_DIR

### Parameters
# Files
CALIBRATION_FNS = {
    "LinearCal": STEP_2_DIR / "calibration.csv",
}


LAYOUT = Square(6.0, 5).rotate(45).rotate(-2.5)

# Controller optimisers
CONTROLLERS = {
    "nocontrol": NoControl,
    "yawcontrol": YawControl,
    "thrustcontrol": ThrustControl,
    "jointcontrol": JointControl,
}

CASE_NOTE = "calibrated_on_diamond"


def run(calibration_key: str, output_path: Path) -> None:
    # Read calibration from calibration file

    with open(CALIBRATION_FNS[calibration_key], "r") as f:
        out = f.read()
        a, b, c = [float(x) for x in out.split(",")]

    # Initialise MITWindfarm using VariableKwGaussianWakeModel
    windfarm = Windfarm(
        rotor_model=UnifiedLUTAD(),
        wake_model=VariableKwGaussianWakeModel(a, b, c, x0=3.0),
        TIamb=TIAMB,
        superposition=Niayifar(),
    )
    # For each controller to optimise...
    for name, controller in CONTROLLERS.items():
        filestem = f"MITWindfarm_wdir-2.5_{calibration_key}_{name}"

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
    run("LinearCal", STEP_3_DIR)
