import copy
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import joblib
import numpy as np
import pandas as pd

from compare import validate_runs
from uwb.config import N_TAPS
from uwb.dataset import (build_feature_table, dataset_fingerprint, feature_columns,
                         load_recordings, save_dataset_csv)
from uwb.features import extract_features
from uwb.models import MODULES
from uwb.preprocess import sliding_windows
from uwb.realtime import PresenceDetector
from uwb.training import (INCOMPLETE_FILE, _fit, cross_validate, environment_versions,
                          parse_args, run, source_fingerprint)


ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "ml_dataset.csv"
        rng = np.random.default_rng(1)
        self.sessions = [
            (f"s{i}", label,
             (rng.normal(size=(600, N_TAPS)) + 1j * rng.normal(size=(600, N_TAPS))) * 10)
            for i, label in enumerate(["fuera_zona", "vacia"])
        ]
        save_dataset_csv(self.path, self.sessions, 20)

    def edit_csv(self, edit):
        table = pd.read_csv(self.path)
        edit(table)
        table.to_csv(self.path, index=False)

    def test_load_and_windows(self):
        recs = load_recordings(self.path)
        self.assertEqual([r["session"] for r in recs], ["s0", "s1"])
        self.assertAlmostEqual(recs[0]["frame_rate"], 20)
        self.assertEqual(len(list(sliding_windows(600, recs[0]["frame_rate"], 6, 1))), 25)

    def test_feature_table(self):
        table = build_feature_table(self.path)
        self.assertEqual(len(table), 50)
        self.assertEqual(set(table["label"]), {"inside", "outside"})
        self.assertEqual(len(feature_columns(table)), 37)
        self.assertTrue(np.isfinite(table[feature_columns(table)].to_numpy()).all())

    def test_presence_column_is_ignored(self):
        self.edit_csv(lambda t: t.__setitem__("presence", 1))
        table = build_feature_table(self.path)
        self.assertEqual(set(table["label"]), {"inside", "outside"})

    def test_link_columns_are_used_when_present(self):
        n = len(pd.read_csv(self.path))
        self.edit_csv(lambda t: (t.__setitem__("dist_cm", 200.0 + np.arange(n) % 7),
                                 t.__setitem__("aoa_deg", 10.0)))
        table = build_feature_table(self.path)
        self.assertTrue((table["link_frac"] == 1).all())
        self.assertTrue((table["dist_range_m"] > 0).all())
        self.assertTrue((table["aoa_std_deg"] == 0).all())
        self.assertTrue(np.isfinite(load_recordings(self.path)[0]["link"]).all())

    def test_missing_link_gives_zero_features(self):
        table = build_feature_table(self.path)
        self.assertTrue((table["link_frac"] == 0).all())
        self.assertTrue(np.isfinite(table[feature_columns(table)].to_numpy()).all())

    def test_lost_frames_warn(self):
        self.edit_csv(lambda t: t.__setitem__("idx", t["idx"] + (t.index >= 10)))
        with self.assertWarnsRegex(UserWarning, "tramas perdidas"):
            self.assertEqual(load_recordings(self.path)[0]["lost_frames"], 1)

    def test_non_monotonic_idx(self):
        self.edit_csv(lambda t: t.__setitem__("idx", 1))
        with self.assertRaises(ValueError):
            load_recordings(self.path)

    def test_invalid_cir(self):
        self.edit_csv(lambda t: t.__setitem__("rx1_i_0", np.nan))
        with self.assertRaisesRegex(ValueError, "CIR"):
            load_recordings(self.path)

    def test_mixed_labels_message(self):
        self.edit_csv(lambda t: t.__setitem__(
            "scenario", np.where(t.index % 2, "fuera_zona", "vacia")))
        with self.assertRaisesRegex(ValueError, "mezclar"):
            build_feature_table(self.path)

    def test_unknown_labels(self):
        self.edit_csv(lambda t: t.__setitem__("scenario", "escenario"))
        with self.assertRaisesRegex(ValueError, "desconocidos"):
            build_feature_table(self.path)

    def test_fingerprint_detects_content_changes(self):
        before = dataset_fingerprint(self.path)
        self.edit_csv(lambda t: t.__setitem__("rx1_i_0", t["rx1_i_0"] + 1))
        self.assertNotEqual(before, dataset_fingerprint(self.path))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = pd.DataFrame({"all_peak_db": rng.normal(size=48)})
        self.y = np.tile(["inside", "outside"], 24)
        self.groups = np.repeat(np.arange(4), 12)

    def test_invalid_window_parameters(self):
        for rate, window, hop in [(0, 6, 1), (20, -1, 1), (20, 6, 0),
                                  (np.nan, 6, 1), (20, 0.01, 1)]:
            with self.subTest(values=(rate, window, hop)):
                with self.assertRaises(ValueError):
                    list(sliding_windows(600, rate, window, hop))

    def test_tune_requires_inner_groups(self):
        with self.assertRaisesRegex(ValueError, "al menos 3 grupos"):
            cross_validate(MODULES["umbral"], self.X, self.y,
                           np.repeat([0, 1], 24), 2, True, False)

    def test_single_class_inner_training(self):
        y = np.repeat(["inside", "inside", "outside"], 16)
        with self.assertRaisesRegex(ValueError, "una clase"):
            _fit(MODULES["umbral"], self.X, y, np.repeat([0, 1, 2], 16), True)

    def test_tuning_valid_groups(self):
        module = SimpleNamespace(NAME="umbral", build=MODULES["umbral"].build,
                                 GRID={"feature": ["all_peak_db"]})
        model, params = _fit(module, self.X, self.y, self.groups, True)
        self.assertEqual(params, {"feature": "all_peak_db"})
        self.assertEqual(len(model.predict(self.X)), len(self.y))

    def test_svm_grouped_calibration(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            model, _ = _fit(MODULES["svm"], self.X, self.y, self.groups, False)
        for train, test in model.cv:
            self.assertTrue(set(self.groups[train]).isdisjoint(self.groups[test]))
        self.assertTrue(np.isfinite(model.predict_proba(self.X)).all())
        np.testing.assert_allclose(model.predict_proba(self.X).sum(axis=1), 1)

    def test_failed_run_marks_existing_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / "gb"
            out.mkdir()
            previous = '{"previous": true}'
            (out / "metrics.json").write_text(previous)
            with self.assertRaisesRegex(SystemExit, "Error en los datos"):
                run("gb", ["--data", str(root / "missing"), "--out", str(root)])
            self.assertTrue((out / INCOMPLETE_FILE).exists())
            self.assertEqual((out / "metrics.json").read_text(), previous)
            with patch("sys.argv", ["compare.py", "--results", str(root)]):
                with self.assertRaisesRegex(SystemExit, "fallidos o en curso"):
                    runpy.run_path(str(ROOT / "compare.py"), run_name="__main__")

    def test_out_equals_forwarded_by_train_all(self):
        with patch("sys.argv", ["train_all.py", "--out=custom results"]), patch(
            "subprocess.run"
        ) as execute:
            runpy.run_path(str(ROOT / "train_all.py"), run_name="__main__")
        self.assertEqual(execute.call_args_list[-1].args[0][-2:],
                         ["--results", "custom results"])

    def test_invalid_folds(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args("gb", ["--folds", "1"])

    def test_comparison_rejects_stale_and_mismatched_results(self):
        run_info = {
            "data_fingerprint": "abc", "source_fingerprint": source_fingerprint(),
            "environment": environment_versions(), "data_path": "unused",
            "group": "room", "folds": 5, "tuned": False, "n_windows": 1500,
        }
        with patch("compare.dataset_fingerprint", return_value="abc"):
            validate_runs([run_info, copy.deepcopy(run_info)])
            for key, value in [("folds", 3), ("tuned", True), ("data_fingerprint", "changed")]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_runs([run_info, {**run_info, key: value}])
            with self.assertRaisesRegex(ValueError, "antiguos"):
                validate_runs([{}])
            with self.assertRaisesRegex(ValueError, "código"):
                validate_runs([{**run_info, "source_fingerprint": "old"}])
        with patch("compare.dataset_fingerprint", return_value="modified"):
            with self.assertRaisesRegex(ValueError, "Han cambiado los datos"):
                validate_runs([run_info])


class RealtimeTests(unittest.TestCase):
    def test_block_selection_copy_and_window_parity(self):
        rng = np.random.default_rng(2)
        cir = rng.normal(size=(140, N_TAPS)) + 1j * rng.normal(size=(140, N_TAPS))
        link = np.column_stack([200 + np.arange(len(cir)) % 5, np.full(len(cir), 10.0)])
        features = extract_features(cir[:120], 20, link[:120])
        columns = list(features)
        module = MODULES["umbral"]
        X = pd.DataFrame([features, {**features, "all_peak_db": features["all_peak_db"] + 10}])
        model = module.build().fit(X, ["outside", "inside"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump({"model": model, "features": columns, "rx": "rx1",
                         "window_s": 6, "hop_s": 1}, path)
            detector = PresenceDetector(path, 20)
            frame = np.zeros(N_TAPS, dtype=complex)
            decisions = []
            for i, values in enumerate(cir):
                frame[:] = values
                decision = detector.push(frame, 200 + i % 5, 10.0)
                if decision is not None:
                    decisions.append(i)
                    start = i + 1 - 120
                    expected = pd.DataFrame([extract_features(cir[start:i + 1], 20, link[start:i + 1])])[columns]
                    self.assertEqual(decision[0], model.predict(expected)[0])
                    self.assertAlmostEqual(
                        decision[1], model.predict_proba(expected)[0, 0]
                    )
            self.assertEqual(decisions, [119, 139])
            with self.assertRaisesRegex(ValueError, "no finitos"):
                detector.push(np.full(N_TAPS, np.nan))


if __name__ == "__main__":
    unittest.main()
