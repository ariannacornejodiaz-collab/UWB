"""Entrenamiento y evaluación de un detector, común a train_rf.py, train_gb.py, etc.

La validación deja fuera sesiones enteras (el proyecto usa una sola sala): nunca hay
ventanas de la misma grabación en entrenamiento y en test, porque si no el acierto
sale inflado.

Cada ejecución guarda en results/<modelo>/:
  metrics.json        métricas, acierto por escenario, configuración usada
  predictions.csv     la predicción de validación de cada ventana
  confusion.png       matriz de confusión
  importance.csv/png  importancia por permutación de cada característica
  model.joblib        el modelo entrenado con todos los datos, para tiempo real
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold

from . import config, plots
from .dataset import build_feature_table, dataset_fingerprint, feature_columns
from .models import MODULES

RESULTS_DIR = Path("results")
INCOMPLETE_FILE = "training-incomplete.txt"


def source_fingerprint():
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_versions():
    return {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}


def parse_args(name, argv=None):
    module = MODULES[name]
    parser = argparse.ArgumentParser(description=f"Entrena y evalúa {module.TITLE}")
    parser.add_argument("--data", default="data/synthetic", help="ml_dataset.csv o carpeta con CSV de ese formato")
    parser.add_argument("--folds", type=int, default=5, help="número máximo de pliegues")
    parser.add_argument("--tune", action="store_true", help="búsqueda de hiperparámetros anidada")
    parser.add_argument("--no-importance", action="store_true", help="no calcula la importancia")
    parser.add_argument("--out", default=str(RESULTS_DIR), help="carpeta de resultados")
    args = parser.parse_args(argv)
    if args.folds < 2:
        parser.error("--folds debe ser al menos 2")
    return args


def grouped_splits(X, y, groups, n_splits, context):
    if n_splits < 2:
        raise ValueError(f"{context}: hacen falta al menos 2 grupos de entrenamiento; "
                         "para --tune o SVM se necesitan al menos 3 grupos en total")
    splits = list(GroupKFold(n_splits).split(X, y, groups))
    for fold, (train, _) in enumerate(splits):
        if len(np.unique(y[train])) < 2:
            raise ValueError(f"{context}, pliegue {fold}: solo hay una clase en entrenamiento; "
                             "hacen falta más grabaciones de cada clase repartidas entre los grupos")
    return splits


def _fit(module, X, y, groups, tune):
    """Entrena el modelo; con tune, elige hiperparámetros dejando fuera grupos enteros."""
    estimator, params = clone(module.build()), None
    splits = None
    if tune or module.NAME == "svm":
        splits = grouped_splits(X, y, groups, min(3, len(np.unique(groups))),
                                "Validación interna" if tune else "Calibración SVM")
    if tune:
        search = GridSearchCV(estimator, module.GRID, cv=splits, scoring="balanced_accuracy",
                              n_jobs=-1, error_score="raise")
        search.fit(X, y)
        estimator, params = search.best_estimator_, search.best_params_
    if module.NAME == "svm":
        # El escalado y la calibración también dejan grupos completos fuera.
        estimator = CalibratedClassifierCV(estimator, cv=splits, ensemble=False).fit(X, y)
    elif not tune:
        estimator.fit(X, y)
    return estimator, params


def cross_validate(module, X, y, groups, n_splits, tune, importance):
    """Predicción de cada ventana hecha por un modelo que no vio su grupo."""
    pred = np.empty(len(y), dtype=object)
    proba = np.full(len(y), np.nan)
    importances, best_params = [], []
    splits = grouped_splits(X, y, groups, n_splits, "Validación externa")
    if tune or module.NAME == "svm":
        for train, _ in splits:
            grouped_splits(X.iloc[train], y[train], groups[train],
                           min(3, len(np.unique(groups[train]))), "Validación interna")
    for train, test in splits:
        model, params = _fit(module, X.iloc[train], y[train], groups[train], tune)
        best_params.append(params)
        pred[test] = model.predict(X.iloc[test])
        if hasattr(model, "predict_proba"):
            column = list(model.classes_).index(config.POSITIVE)
            proba[test] = model.predict_proba(X.iloc[test])[:, column]
        if importance:
            result = permutation_importance(model, X.iloc[test], y[test], scoring="accuracy",
                                            n_repeats=3, random_state=0, n_jobs=-1)
            importances.append(result.importances_mean)
    ranking = None
    if importances:
        ranking = pd.Series(np.mean(importances, axis=0), index=X.columns).sort_values(ascending=False)
    return pred, proba, ranking, best_params


def compute_metrics(table, y, pred, proba):
    inside, pred_inside = y == config.POSITIVE, pred == config.POSITIVE
    correct = pd.Series(pred == y, index=table.index)
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision_inside": precision_score(y, pred, pos_label=config.POSITIVE, zero_division=0),
        "recall_inside": recall_score(y, pred, pos_label=config.POSITIVE, zero_division=0),
        "f1_inside": f1_score(y, pred, pos_label=config.POSITIVE, zero_division=0),
        # Falsa alarma: decir "inside" sin nadie dentro. No detectado: no ver a quien está dentro
        "false_alarm_rate": float(np.mean(pred_inside[~inside])) if (~inside).any() else float("nan"),
        "miss_rate": float(np.mean(~pred_inside[inside])) if inside.any() else float("nan"),
        "roc_auc": float("nan"),
    }
    if np.isfinite(proba).all() and len(np.unique(y)) == 2:
        metrics["roc_auc"] = roc_auc_score(inside, proba)
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    metrics["by_scenario"] = correct.groupby(table["scenario"]).mean().round(4).to_dict()
    return metrics


def print_report(title, metrics, matrix):
    print(f"\n=== {title} ===")
    print(f"exactitud {metrics['accuracy']:.3f} | exactitud equilibrada {metrics['balanced_accuracy']:.3f}"
          f" | F1 inside {metrics['f1_inside']:.3f} | AUC {metrics['roc_auc']:.3f}")
    print(f"falsas alarmas (dice inside sin nadie dentro) {metrics['false_alarm_rate']:.3f}"
          f" | inside no detectado {metrics['miss_rate']:.3f}")
    print(pd.DataFrame(matrix, index=[f"real {c}" for c in config.CLASSES],
                       columns=[f"pred {c}" for c in config.CLASSES]))
    print("acierto por escenario:")
    for scenario, value in metrics["by_scenario"].items():
        print(f"  {scenario:<22} {value:.3f}")


def run(name, argv=None):
    args = parse_args(name, argv)
    module = MODULES[name]
    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    incomplete = out_dir / INCOMPLETE_FILE
    incomplete.write_text(
        f"Entrenamiento iniciado {time.strftime('%Y-%m-%d %H:%M:%S')}, datos: {args.data}\n"
        "No uses los resultados anteriores hasta completar un nuevo entrenamiento.\n",
        encoding="utf-8",
    )

    t0 = time.time()
    try:
        table = build_feature_table(args.data)
        data_hash = dataset_fingerprint(args.data)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error en los datos: {error}")
    columns = feature_columns(table)
    X, y = table[columns], table["label"].to_numpy()

    classes = sorted(set(y))
    if len(classes) < 2:
        raise SystemExit(f"Todas las ventanas de {args.data} son '{classes[0]}'. Para entrenar hacen "
                         "falta grabaciones de las dos clases: con alguien dentro, y con la sala vacía "
                         "o la persona fuera.")
    group = "session"   # una sola sala: se deja fuera cada sesión entera
    groups = table[group].to_numpy()
    n_splits = min(args.folds, len(np.unique(groups)))
    if n_splits < 2:
        raise SystemExit(f"Hacen falta al menos 2 valores distintos de '{group}' para validar")

    print(f"{module.TITLE}: {len(table)} ventanas, {table['session'].nunique()} sesiones, "
          f"{len(columns)} características "
          f"({time.time() - t0:.1f} s en leer y extraer)")
    print("ventanas por clase:", table["label"].value_counts().to_dict())
    print(f"validación: {n_splits} pliegues dejando fuera '{group}'" + (" con --tune" if args.tune else ""))

    t0 = time.time()
    try:
        pred, proba, ranking, fold_params = cross_validate(
            module, X, y, groups, n_splits, args.tune, not args.no_importance)
    except ValueError as error:
        raise SystemExit(f"Error en la validación: {error}")
    cv_seconds = time.time() - t0

    metrics = compute_metrics(table, y, pred, proba)
    matrix = confusion_matrix(y, pred, labels=config.CLASSES)
    print_report(module.TITLE, metrics, matrix)

    # Modelo final con todos los datos, el que se usa en tiempo real
    final, best_params = _fit(module, X, y, groups, args.tune)
    t0 = time.time()
    final.predict(X)
    ms_per_window = 1000 * (time.time() - t0) / len(X)
    joblib.dump({"model": final, "features": columns, "name": name, "title": module.TITLE,
                 "window_s": config.WINDOW_S, "hop_s": config.HOP_S, "rx": config.RX},
                out_dir / "model.joblib")

    summary = {
        "name": name,
        "title": module.TITLE,
        "data": args.data,
        "data_path": str(Path(args.data).resolve()),
        "data_fingerprint": data_hash,
        "source_fingerprint": source_fingerprint(),
        "environment": environment_versions(),
        "group": group,
        "folds": n_splits,
        "tuned": args.tune,
        "n_windows": int(len(table)),
        "n_sessions": int(table["session"].nunique()),
        "windows_per_class": {k: int(v) for k, v in table["label"].value_counts().items()},
        **metrics,
        "confusion_matrix": {"labels": config.CLASSES, "values": matrix.tolist()},
        "best_params": best_params,
        "fold_params": fold_params if args.tune else None,
        "cv_seconds": round(cv_seconds, 1),
        "ms_per_window": round(ms_per_window, 3),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    predictions = table[["session", "scenario", "t_start_s", "label"]].copy()
    predictions["pred"], predictions["p_inside"] = pred, proba
    predictions.to_csv(out_dir / "predictions.csv", index=False)

    plots.confusion(matrix, config.CLASSES, f"{module.TITLE} — validación dejando fuera '{group}'",
                    out_dir / "confusion.png")
    if ranking is not None:
        ranking.round(4).rename("importance").to_csv(out_dir / "importance.csv")
        plots.importance(ranking, f"{module.TITLE} — importancia por permutación", out_dir / "importance.png")
        print("\ncaracterísticas más importantes (caída de exactitud al desordenarlas):")
        print(ranking.head(8).round(4).to_string())

    if best_params:
        print("\nhiperparámetros elegidos:", best_params)
    if ranking is None:
        for filename in ["importance.csv", "importance.png"]:
            (out_dir / filename).unlink(missing_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    incomplete.unlink()
    print(f"\nvalidación en {cv_seconds:.1f} s | {ms_per_window:.3f} ms por ventana en predicción")
    print(f"resultados en {out_dir}/")
    return summary
