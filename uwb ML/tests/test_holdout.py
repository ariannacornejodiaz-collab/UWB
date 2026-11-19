from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from evaluate_holdout import evaluate, validate_holdout
import pandas as pd
from uwb.simulator import generate_dataset
from uwb.training import run


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = self.root / "train"
        self.test = self.root / "test"
        generate_dataset(self.train, reps=2, duration_s=7, seed=1)
        generate_dataset(self.test, reps=1, duration_s=7, seed=2, name_prefix="holdout_")

    def test_independent_sets_accepted(self):
        validate_holdout(self.train, self.test)

    def test_training_dataset_rejected(self):
        with self.assertRaisesRegex(ValueError, "entrenamiento"):
            validate_holdout(self.train, self.train)

    def test_shared_sessions_rejected(self):
        train = pd.read_csv(self.train / "ml_dataset.csv")
        test = pd.read_csv(self.test / "ml_dataset.csv")
        test.loc[test["session"] == test["session"].iloc[0], "session"] = train["session"].iloc[0]
        test.to_csv(self.test / "ml_dataset.csv", index=False)
        with self.assertRaisesRegex(ValueError, "session"):
            validate_holdout(self.train, self.test)

    def test_renamed_recording_rejected(self):
        train = pd.read_csv(self.train / "ml_dataset.csv")
        first = train[train["session"] == train["session"].iloc[0]].copy()
        first["session"] = "renamed"
        test = pd.read_csv(self.test / "ml_dataset.csv")
        pd.concat([test, first]).to_csv(self.test / "ml_dataset.csv", index=False)
        with self.assertRaisesRegex(ValueError, "idénticas"):
            validate_holdout(self.train, self.test)

    def test_evaluation_does_not_fit_or_modify_saved_model(self):
        models = self.root / "models"
        out = self.root / "evaluation"
        with redirect_stdout(io.StringIO()):
            run("umbral", ["--data", str(self.train), "--out", str(models),
                           "--no-importance"])
        model = models / "umbral" / "model.joblib"
        before = hashlib.sha256(model.read_bytes()).hexdigest()
        with patch("uwb.models.threshold.ThresholdDetector.fit",
                   side_effect=AssertionError("No debe entrenar")), redirect_stdout(io.StringIO()):
            reports = evaluate(models, self.test, out)
        self.assertEqual(before, hashlib.sha256(model.read_bytes()).hexdigest())
        self.assertFalse(reports[0]["trained_during_evaluation"])
        self.assertEqual(reports[0]["n_windows"], 10)
        self.assertTrue((out / "umbral" / "predictions.csv").exists())
        self.assertTrue((out / "comparison.png").exists())
        self.assertFalse((out / "evaluation-incomplete.txt").exists())
        self.assertFalse(list(out.rglob("model.joblib")))
        with self.assertRaisesRegex(ValueError, "fuera de la carpeta"):
            evaluate(models, self.test, models)

    def test_invalid_generation_parameters_rejected(self):
        for args in [{"reps": 0}, {"duration_s": 0}, {"name_prefix": "..\\outside"}]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                generate_dataset(self.root / "invalid", **args)


if __name__ == "__main__":
    unittest.main()
