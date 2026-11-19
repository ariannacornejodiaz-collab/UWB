"""Evalúa modelos guardados en sesiones nuevas de la misma sala; nunca entrena ni ajusta modelos."""
import argparse
import json
from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

from compare import validate_runs
from uwb import config, plots
from uwb.dataset import (build_feature_table, dataset_fingerprint, feature_columns,
                         load_recordings, session_hashes)
from uwb.training import INCOMPLETE_FILE, compute_metrics, print_report


def validate_holdout(train_dir, test_dir):
    train_dir, test_dir = Path(train_dir).resolve(), Path(test_dir).resolve()
    if train_dir == test_dir:
        raise ValueError("El conjunto de prueba no puede ser el de entrenamiento")
    train, test = load_recordings(train_dir), load_recordings(test_dir)
    overlap = {r["session"] for r in train} & {r["session"] for r in test}
    if overlap:
        raise ValueError(f"Hay session compartidos con entrenamiento: {sorted(overlap)}")
    if session_hashes(train_dir) & session_hashes(test_dir):
        raise ValueError("Hay grabaciones idénticas a las de entrenamiento, aunque renombradas")

def evaluate(results_dir, data_dir, out_dir):
    results_dir, data_dir, out_dir = map(Path, [results_dir, data_dir, out_dir])
    if (out_dir.resolve() == results_dir.resolve()
            or results_dir.resolve() in out_dir.resolve().parents
            or out_dir.resolve() in results_dir.resolve().parents):
        raise ValueError("La evaluación debe guardarse fuera de la carpeta de modelos")
    if list(results_dir.glob(f"*/{INCOMPLETE_FILE}")):
        raise ValueError("Hay entrenamientos incompletos; complétalos antes de evaluar")
    metric_files = sorted(results_dir.glob("*/metrics.json"))
    if not metric_files:
        raise ValueError(f"No hay modelos entrenados en {results_dir}")
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in metric_files]
    validate_runs(runs)
    for train_dir in {r["data_path"] for r in runs}:
        validate_holdout(train_dir, data_dir)

    table = build_feature_table(data_dir)
    y = table["label"].to_numpy()
    if set(y) != set(config.CLASSES):
        raise ValueError("La prueba debe contener inside y outside para medir ambos errores")
    data_hash = dataset_fingerprint(data_dir)
    bundles = []
    for path, run in zip(metric_files, runs):
        bundle = joblib.load(path.parent / "model.joblib")
        if (bundle["name"] != run["name"]
                or bundle["window_s"] != config.WINDOW_S
                or bundle["hop_s"] != config.HOP_S
                or bundle.get("rx") != config.RX
                or bundle["features"] != feature_columns(table)):
            raise ValueError(f"Modelo o preprocesado incompatible: {path.parent}")
        bundles.append(bundle)

    out_dir.mkdir(parents=True, exist_ok=True)
    incomplete = out_dir / "evaluation-incomplete.txt"
    incomplete.write_text("Evaluación en curso o fallida; no usar resultados previos.\n",
                          encoding="utf-8")
    summaries = []
    for path, run, bundle in zip(metric_files, runs, bundles):
        model = bundle["model"]
        X = table[bundle["features"]]
        pred = model.predict(X)
        positive = list(model.classes_).index(config.POSITIVE)
        proba = model.predict_proba(X)[:, positive]
        metrics = compute_metrics(table, y, pred, proba)
        matrix = confusion_matrix(y, pred, labels=config.CLASSES)
        print_report(bundle["title"] + " — prueba independiente", metrics, matrix)
        summary = {
            "name": run["name"], "title": bundle["title"], "evaluation": "holdout",
            "trained_during_evaluation": False, "model_path": str((path.parent / "model.joblib").resolve()),
            "training_data_fingerprint": run["data_fingerprint"],
            "test_data": str(data_dir.resolve()), "test_data_fingerprint": data_hash,
            "n_windows": len(table),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), **metrics,
            "confusion_matrix": {"labels": config.CLASSES, "values": matrix.tolist()},
            "limitation": "Solo evaluación sintética si los datos son simulados; no valida el radar real",
        }
        target = out_dir / run["name"]
        target.mkdir(exist_ok=True)
        (target / "metrics.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        predictions = table[["session", "scenario", "t_start_s", "label"]].copy()
        predictions["pred"], predictions["p_inside"] = pred, proba
        predictions["correct"] = pred == y
        predictions.to_csv(target / "predictions.csv", index=False)
        plots.confusion(matrix, config.CLASSES, bundle["title"] + " — holdout",
                        target / "confusion.png")
        summaries.append(summary)
    columns = ["name", "accuracy", "balanced_accuracy", "f1_inside", "false_alarm_rate", "miss_rate"]
    comparison = pd.DataFrame(summaries).set_index("name")
    comparison[[c for c in columns if c != "name"]].to_csv(out_dir / "comparison.csv")
    plots.comparison(comparison, ["balanced_accuracy", "recall_inside", "precision_inside"],
                     ["Exactitud equilibrada", "Detecta dentro", "Precisión dentro"],
                     out_dir / "comparison.png")
    incomplete.unlink()
    print(f"\nResultados independientes: {out_dir}")
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="results")
    parser.add_argument("--data", default=str(Path("data") / "synthetic_holdout" / "ml_dataset.csv"))
    parser.add_argument("--out", default="results_holdout")
    args = parser.parse_args(argv)
    try:
        evaluate(args.models, args.data, args.out)
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"No se pudo evaluar: {error}")


if __name__ == "__main__":
    main()
