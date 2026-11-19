"""Compara los detectores ya entrenados leyendo results/*/metrics.json.

    .venv/bin/python compare.py
    .venv/bin/python compare.py --results results

Avisa si los resultados no son comparables (otros datos, otra validación).
Guarda results/comparison.csv y results/comparison.png.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from uwb import plots
from uwb.dataset import dataset_fingerprint
from uwb.training import INCOMPLETE_FILE, environment_versions, source_fingerprint

COLUMNS = {
    "accuracy": "exactitud",
    "balanced_accuracy": "exact. equilibrada",
    "f1_inside": "F1 inside",
    "roc_auc": "AUC",
    "false_alarm_rate": "falsas alarmas",
    "miss_rate": "inside no detectado",
    "ms_per_window": "ms/ventana",
}
CHART = ["balanced_accuracy", "f1_inside", "recall_inside", "precision_inside"]
CHART_LABELS = ["Exactitud\nequilibrada", "F1 inside", "Detecta a quien\nestá dentro", "Acierta cuando\ndice inside"]


def validate_runs(runs):
    keys = ["data_fingerprint", "source_fingerprint", "environment",
            "group", "folds", "tuned", "n_windows"]
    for run in runs:
        if any(key not in run for key in keys) or "data_path" not in run:
            raise ValueError("Hay resultados antiguos sin huellas verificables. "
                             "Vuelve a entrenar los detectores con el código actual.")
    if any(any(run[key] != runs[0][key] for key in keys) for run in runs[1:]):
        raise ValueError("Los resultados usan distintos datos, código, versiones o validación "
                         "(group, folds o tuned). Vuelve a entrenar con las mismas opciones.")
    if runs[0]["source_fingerprint"] != source_fingerprint():
        raise ValueError("El código ha cambiado desde el entrenamiento. Vuelve a entrenar.")
    if runs[0]["environment"] != environment_versions():
        raise ValueError("Las versiones han cambiado desde el entrenamiento. Vuelve a entrenar.")
    for data_path in {run["data_path"] for run in runs}:
        if dataset_fingerprint(data_path) != runs[0]["data_fingerprint"]:
            raise ValueError(f"Han cambiado los datos de {data_path}. Vuelve a entrenar.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    incomplete = sorted(Path(args.results).glob(f"*/{INCOMPLETE_FILE}"))
    if incomplete:
        raise SystemExit("Hay entrenamientos fallidos o en curso: "
                         + ", ".join(str(p.parent) for p in incomplete)
                         + ". Completa esos entrenamientos antes de comparar.")
    files = sorted(Path(args.results).glob("*/metrics.json"))
    if not files:
        raise SystemExit(f"No hay resultados en {args.results}/. Ejecuta antes train_rf.py, "
                         "train_gb.py, train_svm.py y train_umbral.py")
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    try:
        validate_runs(runs)
    except (ValueError, OSError) as error:
        raise SystemExit(f"No se pueden comparar los resultados: {error}")
    order = {name: i for i, name in enumerate(plots.DETECTOR_ORDER)}
    runs.sort(key=lambda r: order.get(r["name"], len(order)))
    summary = pd.DataFrame(runs).set_index("name")

    row = runs[0]
    print(f"datos: {row['data']} | {row['n_windows']} ventanas, {row['n_sessions']} sesiones (una sola sala) | validación dejando fuera '{row['group']}', "
          f"{row['folds']} pliegues\n")
    print("Fecha de cada entrenamiento:")
    print(summary[["created"]].to_string(), "\n")

    table = summary[["title", "tuned"] + list(COLUMNS)].rename(columns=COLUMNS)
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(table.to_string())

    scenarios = pd.DataFrame({r["title"]: r["by_scenario"] for r in runs})
    print("\nacierto por escenario:")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(scenarios.to_string())

    ml = summary[summary.index != "umbral"]
    best = ml["balanced_accuracy"].idxmax() if len(ml) else None
    if best is not None:
        print(f"\nmejor modelo de ML por exactitud equilibrada: {summary.loc[best, 'title']} "
              f"({summary.loc[best, 'balanced_accuracy']:.3f})")
        if "umbral" in summary.index:
            gain = summary.loc[best, "balanced_accuracy"] - summary.loc["umbral", "balanced_accuracy"]
            print(f"mejora sobre el umbral: {gain * 100:+.1f} puntos")
        print("Mira también las falsas alarmas, el acierto por escenario y si las diferencias "
              "son mayores que la variación entre sesiones antes de decidir.")

    out = Path(args.results)
    table.join(scenarios.T.add_prefix("escenario "), on="title").to_csv(out / "comparison.csv")
    plots.comparison(summary, CHART, CHART_LABELS, out / "comparison.png")
    print(f"\ntabla en {out / 'comparison.csv'} y gráfica en {out / 'comparison.png'}")
