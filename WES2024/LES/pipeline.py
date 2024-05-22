from itertools import product
from pathlib import Path
import json
import luigi
import run as backend
import polars as pl

WDIRS = [-2.5, 0.0, 42.0, 45.0]
CONTROLLERS = ["nocontrol", "yawcontrol", "thrustcontrol", "jointcontrol"]

DATA_DIR = Path(__file__).parent


class Calibration(luigi.Task):
    wdir = luigi.FloatParameter()

    def requires(self):
        return LES(self.wdir, "nocontrol")

    def run(self):
        df = pl.read_csv(self.requires().output().path)

        calibration = backend.CalibrationCase(self.wdir, backend.TIAMB, backend.ROW_INDICES)
        calibration = calibration.calibrate(df["Cp"].to_numpy())
        output = ",".join(str(x) for x in calibration.setpoints())

        # Make directory if it does not exist.
        Path(self.output().path).parent.mkdir(exist_ok=True, parents=True)

        # Write calibration values to file.
        with open(self.output().path, "w") as f:
            f.write(output)

    def output(self):
        return luigi.LocalTarget(DATA_DIR / f"calibration/calibration_wdir{self.wdir}.csv")


class ZeroSetpoints(luigi.Task):
    wdir = luigi.FloatParameter()

    def run(self):
        calibration = backend.CalibrationCase(self.wdir, backend.TIAMB, backend.ROW_INDICES)
        sol = calibration.run_model(1, 1, 1)
        sim_dict = backend.make_sim_case(sol, self.wdir, "nocontrol")
        with open(self.output().path, "w") as f:
            json.dump(sim_dict, f, indent=4)

    def output(self):
        return luigi.LocalTarget(DATA_DIR / f"LES_input/diamond_wdir{self.wdir}_nocontrol.json")


class Setpoints(luigi.Task):
    wdir = luigi.FloatParameter()
    controller = luigi.Parameter()

    def requires(self):
        if self.controller != "nocontrol":
            raise ValueError(
                "Can't call Setpoints with nocontrol controller. Use ZeroSetpoints instead."
            )
        return Calibration(self.wdir)

    def run(self):
        # TO DO
        ...

        # sim_dict = backend.make_sim_case(sol, self.wdir, self.controller)
        # with open(
        #     DATA_DIR / f"LES_input/diamond_wdir{self.wdir}_{self.controller}.json", "w"
        # ) as f:
        #     json.dump(sim_dict, f, indent=4)

    def output(self):
        return luigi.LocalTarget(
            DATA_DIR / f"LES_input/diamond_wdir{self.wdir}_{self.controller}.json"
        )


class _data_from_LES(luigi.ExternalTask):
    wdir = luigi.FloatParameter()
    controller = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(DATA_DIR / f"LES_output/LES_wdir{self.wdir}_{self.controller}.csv")


class LES(luigi.Task):
    wdir = luigi.FloatParameter()
    controller = luigi.Parameter()

    def requires(self):
        if self.controller == "nocontrol":
            return [ZeroSetpoints(self.wdir), _data_from_LES(self.wdir, self.controller)]
        else:
            return [
                Setpoints(self.wdir, self.controller),
                _data_from_LES(self.wdir, self.controller),
            ]

    def output(self):
        return luigi.LocalTarget(DATA_DIR / f"LES_output/LES_wdir{self.wdir}_{self.controller}.csv")


class Combine(luigi.Task):
    def requires(self):
        params = product(WDIRS, CONTROLLERS)

        return [LES(wdir, cont) for wdir, cont in params]

    def run(self):
        ...

    def output(self):
        return luigi.LocalTarget("combined_output.csv")


if __name__ == "__main__":
    luigi.build([Combine()], local_scheduler=False)
