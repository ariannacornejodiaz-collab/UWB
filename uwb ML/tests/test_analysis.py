import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze_dataset import analyze_dataset, main
from uwb.config import N_TAPS
from uwb.dataset import save_dataset_csv


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rng = np.random.default_rng(4)

    def write(self, scenarios):
        sessions = [(f"s{i}", s, self.rng.normal(size=(140, N_TAPS))
                     + 1j * self.rng.normal(size=(140, N_TAPS)))
                    for i, s in enumerate(scenarios)]
        path = self.root / "ml_dataset.csv"
        save_dataset_csv(path, sessions, 20)
        return path

    def test_missing_classes_block_training_not_analysis(self):
        report, windows = analyze_dataset(self.write(["fuera_zona", "fuera_zona"]))
        self.assertFalse(report["ready_for_training"])
        self.assertEqual(len(windows), 4)
        self.assertTrue(any("outside" in m for m in report["blockers"]))
        json.dumps(report, allow_nan=False)

    def test_unknown_scenario_is_reported(self):
        report, _ = analyze_dataset(self.write(["escenario", "vacia"]))
        self.assertFalse(report["ready_for_training"])
        self.assertTrue(any("escenario" in m for m in report["blockers"]))

    def test_ready_dataset_and_persistent_reports(self):
        path = self.write(["fuera_zona", "dentro", "vacia"])
        out = self.root / "analysis"
        self.assertEqual(main(["--data", str(path), "--out", str(out)]), 0)
        report = json.loads((out / "dataset_report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["ready_for_training"])
        self.assertTrue((out / "signal_windows.csv").exists())
        self.assertFalse((out / "model.joblib").exists())


if __name__ == "__main__":
    unittest.main()
