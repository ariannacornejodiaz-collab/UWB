"""Audita un CSV de la placa sin inventar etiquetas ni requerir ambas clases."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from uwb.config import CLASSES, HOP_S, MIN_RANGE_M, RX, WINDOW_S, ZERO_TAP, m_to_tap, tap_to_m
from uwb.dataset import load_recordings, window_label
from uwb.preprocess import remove_clutter, sliding_windows


def analyze_dataset(data):
    blockers, notices, recordings, windows = [], [], [], []
    classes = Counter()
    usable_sessions = set()

    for rec in load_recordings(data):
        cir, rate, session = rec["cir"], rec["frame_rate"], rec["session"]
        info = {"file": rec["file"], "session": session, "blockers": [], "notices": []}
        recordings.append(info)
        dt = np.diff(rec["times"])
        median_dt = float(np.median(dt))
        irregular = np.flatnonzero(dt > 1.5 * median_dt)
        rate_windows = list(sliding_windows(len(cir), rate, WINDOW_S, HOP_S))
        first = m_to_tap(MIN_RANGE_M)
        signal_rows, valid, unknown = [], 0, set()
        for start, end in rate_windows:
            try:
                label = window_label(rec["raw_labels"][start:end])
            except ValueError as error:
                unknown.add(str(error))
                continue
            if label is None:
                continue
            valid += 1
            classes[label[1]] += 1
            usable_sessions.add(session)
            power = np.mean(np.abs(remove_clutter(cir[start:end])) ** 2, axis=0)
            peak = first + int(np.argmax(power[first:]))
            floor = float(np.median(power[first:]))
            row = {
                "session": session, "scenario": label[0], "label": label[1],
                "start_frame": start, "end_frame_exclusive": end,
                "t_start_s": float(rec["times"][start]),
                "t_last_frame_s": float(rec["times"][end - 1]),
                "residual_mean_power": float(power[first:].mean()),
                "peak_tap": peak, "peak_range_m_assumed": float(tap_to_m(peak)),
                "peak_over_floor_db": float(10 * np.log10((power[peak] + 1e-12) / (floor + 1e-12))),
            }
            windows.append(row)
            signal_rows.append(row)
        info["blockers"].extend(sorted(unknown))
        if valid == 0:
            info["blockers"].append("No hay ventanas utilizables con un único escenario.")
        if rec["lost_frames"] or len(irregular):
            info["notices"].append(
                "Intervalos de time_epoch irregulares o tramas perdidas (idx): "
                "la FFT supone muestreo uniforme. Revisar antes de remuestrear.")
        early = [r["residual_mean_power"] for r in signal_rows if r["t_start_s"] < 13]
        later = [r["residual_mean_power"] for r in signal_rows if r["t_start_s"] >= 13]
        startup_ratio = float(np.median(early) / max(np.median(later), 1e-12)) if early and later else None
        if startup_ratio is not None and startup_ratio > 10:
            info["notices"].append(
                "Los primeros 13 s tienen mucha más variación que el resto. "
                "Inspeccionar el arranque; no se ha recortado ni atribuido a una causa.")
        info.update({
            "n_frames": len(cir), "elapsed_s": rec["elapsed_s"], "frame_rate_hz": rate,
            "raw_labels": dict(Counter(rec["raw_labels"])),
            "lost_frames": rec["lost_frames"],
            "interval_ms": {name: float(value * 1000) for name, value in zip(
                ["min", "median", "p95", "max"], np.quantile(dt, [0, .5, .95, 1]))},
            "long_intervals": [{"after_frame": int(i), "t_s": float(rec["times"][i]),
                                "interval_ms": float(dt[i] * 1000)} for i in irregular],
            "candidate_windows": len(rate_windows), "homogeneous_windows": valid,
            "coupling_peak_tap": int(np.argmax(np.abs(cir).mean(axis=0))),
            "startup_13s_residual_power_ratio": startup_ratio,
        })
        if info["coupling_peak_tap"] != ZERO_TAP:
            info["notices"].append(
                f"El pico de acoplamiento está en el tap {info['coupling_peak_tap']}, "
                f"no en ZERO_TAP={ZERO_TAP} de config.py.")
        blockers.extend(f"{session}: {m}" for m in info["blockers"])
        notices.extend(f"{session}: {m}" for m in info["notices"])

    missing_classes = sorted(set(CLASSES) - set(classes))
    if missing_classes:
        blockers.append("Faltan ventanas de las clases: " + ", ".join(missing_classes)
                        + ". Añadir sesiones reales con su escenario; no relabelar las existentes.")
    n_groups = len(usable_sessions)
    if n_groups < 2:
        blockers.append("Faltan al menos 2 sesiones utilizables para validación externa.")
    if n_groups < 3:
        notices.append("SVM y --tune requieren al menos 3 sesiones en total, "
                       "con ambas clases en los entrenamientos internos.")
    notices.append("Tener ambas clases globalmente no garantiza que cada pliegue sea válido; "
                   "el entrenamiento comprueba también las clases por sesión.")
    report = {
        "data": str(Path(data).resolve()), "rx": RX,
        "window_s": WINDOW_S, "hop_s": HOP_S,
        "n_recordings": len(recordings), "windows_per_class": dict(classes),
        "suggested_group": "session", "n_usable_groups": n_groups,
        "ready_for_training": not blockers, "blockers": blockers, "notices": notices,
        "recordings": recordings,
        "interpretation": "Potencias residuales sin unidades calibradas. Distancias basadas en "
                          "config.py; no identifican a una persona. "
                          "El informe no entrena ni valida un modelo de presencia.",
    }
    return report, pd.DataFrame(windows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(Path("data") / "real"))
    parser.add_argument("--out", default=str(Path("results_real") / "analysis"))
    args = parser.parse_args(argv)
    try:
        report, windows = analyze_dataset(args.data)
    except (OSError, ValueError) as error:
        raise SystemExit(f"No se pudo analizar el dataset: {error}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    windows.to_csv(out / "signal_windows.csv", index=False)
    print(f"{report['n_recordings']} sesiones; ventanas: {report['windows_per_class']}")
    for message in report["blockers"]:
        print(f"BLOQUEO: {message}")
    for message in report["notices"]:
        print(f"AVISO: {message}")
    print(f"Informe: {out / 'dataset_report.json'}")
    print(f"Diagnóstico por ventana (no características de entrenamiento): {out / 'signal_windows.csv'}")
    return 0 if report["ready_for_training"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
