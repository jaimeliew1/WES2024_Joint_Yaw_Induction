"""
How to run (when `local_scheduler=False`)
^^^^^^^^^^

Run the `luigid` command to initialize the scheduler server.

Access the scheduler dashboard at `http://localhost:8082/` in a web browser.

Run this script (`python WES2024/LES/pipeline.py`)

Refresh the dashboard page.

"""

from itertools import product
from pathlib import Path

import luigi
import WES2024.LES.run as backend


WDIRS = [-2.5]  # , 0.0, 42.0, 45.0]
CONTROLLERS = ["nocontrol", "yawcontrol", "thrustcontrol", "jointcontrol"]

DATA_DIR = Path(__file__).parent


class Calibration(luigi.Task):
    wdir = luigi.FloatParameter()

    def requires(self) -> luigi.Task:
        return [
        # LES(self.wdir, "nocontrol"),
                LES(self.wdir, "thrustcontrolcalibration"),
                ], [
            # ZeroSetpoints(self.wdir),
            Setpoints(self.wdir, "thrustcontrolcalibration"),
        ]

    def run(self):
        LES_res, LES_inputs = self.requires()
        backend.calibrate_wake_model(
            self.wdir,
            [x.output().path for x in LES_res],
            [x.output().path for x in LES_inputs],
            self.output().path,
        )

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(DATA_DIR / f"calibration/calibration_wdir{self.wdir}.csv")


class ZeroSetpoints(luigi.Task):
    wdir = luigi.FloatParameter()

    def run(self):
        backend.no_control_setpoints(self.wdir, self.output().path)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(DATA_DIR / f"LES_input/diamond_wdir{self.wdir}_nocontrol.json")


class Setpoints(luigi.Task):
    wdir = luigi.FloatParameter()
    controller = luigi.Parameter()

    def requires(self):
        if self.controller == "nocontrol":
            raise ValueError(
                "Can't call Setpoints with nocontrol controller. Use ZeroSetpoints instead."
            )
        elif self.controller == "thrustcontrolcalibration":
            raise ValueError(
                "Can't call Setpoints with thrustcontrolcalibration controller. This should be generated manually."
            )

        return Calibration(self.wdir)

    def run(self):
        backend.find_optimal_setpoints(
            self.wdir, self.controller, self.requires().output().path, self.output().path
        )

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


class MITWindfarm(luigi.Task):
    wdir = luigi.FloatParameter()
    controller = luigi.Parameter()

    def requires(self):
        if self.controller == "nocontrol":
            return [Calibration(self.wdir), ZeroSetpoints(self.wdir)]
        else:
            return [Calibration(self.wdir), Setpoints(self.wdir, self.controller)]

    def run(self):
        backend.run_MITWindfarm(
            self.wdir,
            self.controller,
            self.requires()[0].output().path,
            self.requires()[1].output().path,
            self.output().path,
        )

    def output(self):
        return luigi.LocalTarget(
            DATA_DIR / f"LES_output/MITWindfarm_wdir{self.wdir}_{self.controller}.csv"
        )


class Combine(luigi.Task):
    def requires(self):
        params = product(WDIRS, CONTROLLERS)

        return [MITWindfarm(wdir, cont) for wdir, cont in params] + [
            LES(wdir, cont) for wdir, cont in params
        ]

    def run(self):
        backend.combine_results([x.output().path for x in self.requires()], self.output().path)

    def output(self):
        return luigi.LocalTarget("combined_output.csv")


class PlotSingle(luigi.Task):
    wdir = luigi.FloatParameter()
    controller = luigi.Parameter()

    def requires(self):

        return MITWindfarm(self.wdir, self.controller)

    def run(self):
        backend.plot_layout_single(self.requires().output().path, self.output().path)

    def output(self):
        return luigi.LocalTarget(DATA_DIR / f"fig/layout_wdir{self.wdir}_{self.controller}.png")


class Plot(luigi.Task):
    def requires(self):
        params = product(WDIRS, CONTROLLERS)

        return [PlotSingle(wdir, cont) for wdir, cont in params]

    def run(self):
        pass

    def output(self):
        return None


if __name__ == "__main__":
    luigi.build([Combine(), Plot()], local_scheduler=False)
