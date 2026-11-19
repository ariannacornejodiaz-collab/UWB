"""Lectura de los CSV de la placa (formato ml_dataset.csv) y tabla de características.

De las ~580 columnas del CSV solo se usan:
  session, scenario     sesión y escenario (la etiqueta la pone una persona)
  idx, time_epoch       orden y tiempo de cada trama
  rx1_i_*, rx1_q_*      CIR complejo de la antena RX1 (config.RX)
  dist_cm, aoa_deg      distancia y ángulo estimados por la placa (opcionales; vacíos = sin estimación)
No se usan presence, valid, gap_ms, pdoa_* ni rx2_*: presence es la salida del detector
de la placa, que puede ser 0 con una persona quieta, y no es verdad de terreno.
"""
import hashlib
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .config import HOP_S, LABEL_MAP, N_TAPS, RX, WINDOW_S
from .features import extract_features
from .preprocess import sliding_windows

META_COLUMNS = ["session", "raw_label", "label", "scenario", "t_start_s"]
BASE_COLUMNS = ["session", "scenario", "idx", "time_epoch"]
LINK_COLUMNS = ["dist_cm", "aoa_deg"]


def cir_columns(rx=RX):
    return ([f"{rx}_i_{k}" for k in range(N_TAPS)], [f"{rx}_q_{k}" for k in range(N_TAPS)])


def required_columns(rx=RX):
    re, im = cir_columns(rx)
    return BASE_COLUMNS + re + im


def normalize_label(value):
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def is_dataset_csv(path, rx=RX):
    try:
        header = pd.read_csv(path, nrows=0).columns
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    return set(required_columns(rx)) <= set(header)


def dataset_files(data, rx=RX):
    """Un CSV, o los CSV con el formato de la placa dentro de una carpeta."""
    data = Path(data)
    if data.is_file():
        files = [data]
    elif data.is_dir():
        files = sorted(p for p in data.glob("*.csv") if is_dataset_csv(p, rx))
    else:
        raise FileNotFoundError(f"No existe {data}")
    if not files:
        raise ValueError(f"{data}: no hay CSV con el formato de la placa "
                         f"(columnas {BASE_COLUMNS} y {rx}_i_*, {rx}_q_*)")
    return files


def load_recordings(data, rx=RX):
    """Lee las sesiones de un CSV o carpeta. Devuelve una lista de dicts, uno por sesión:

      session, file, cir (complejo [tramas, N_TAPS]), raw_labels (escenario por trama),
      frame_rate (tramas/s según time_epoch), lost_frames (según idx), elapsed_s,
      times (segundos desde la primera trama)
    """
    recordings = []
    re_cols, im_cols = cir_columns(rx)
    for path in dataset_files(data, rx):
        header = pd.read_csv(path, nrows=0).columns
        missing = [c for c in required_columns(rx) if c not in header]
        if missing:
            raise ValueError(f"{path}: faltan {len(missing)} columnas, por ejemplo {missing[:3]}")
        link_cols = [c for c in LINK_COLUMNS if c in header]
        table = pd.read_csv(path, usecols=required_columns(rx) + link_cols)
        for c in LINK_COLUMNS:
            table[c] = pd.to_numeric(table[c], errors="coerce") if c in table else np.nan
        table["session"] = table["session"].astype(str).str.strip()
        for session, part in table.groupby("session", sort=False):
            part = part.sort_values("idx")
            if len(part) < 2:
                raise ValueError(f"{path}, sesión {session}: hacen falta al menos 2 tramas")
            re = part[re_cols].to_numpy(np.float32)
            im = part[im_cols].to_numpy(np.float32)
            if not np.isfinite(re).all() or not np.isfinite(im).all():
                raise ValueError(f"{path}, sesión {session}: el CIR contiene valores vacíos o no finitos")
            idx = pd.to_numeric(part["idx"], errors="raise").to_numpy(float)
            times = pd.to_numeric(part["time_epoch"], errors="raise").to_numpy(float)
            if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
                raise ValueError(f"{path}, sesión {session}: time_epoch debe ser creciente y finito")
            if (idx != np.floor(idx)).any() or (np.diff(idx) <= 0).any():
                raise ValueError(f"{path}, sesión {session}: idx debe ser entero y estrictamente creciente")
            lost = int(np.sum(np.diff(idx) - 1))
            gaps = np.diff(times)
            if lost or gaps.max() > 1.5 * np.median(gaps):
                warnings.warn(
                    f"{path}, sesión {session}: {lost} tramas perdidas o pausas en time_epoch; "
                    "el análisis espectral supone muestreo uniforme. Revisa o recorta la sesión.",
                    UserWarning, stacklevel=2)
            link = part[LINK_COLUMNS].to_numpy(float)
            link[~np.isfinite(link)] = np.nan
            elapsed = float(times[-1] - times[0])
            recordings.append({
                "session": session, "file": path.name,
                "cir": (re + 1j * im).astype(np.complex64), "link": link,
                "raw_labels": part["scenario"].map(normalize_label).to_numpy(),
                "frame_rate": float((len(part) - 1) / elapsed),
                "lost_frames": lost, "elapsed_s": elapsed, "times": times - times[0],
            })
    names = [r["session"] for r in recordings]
    if len(set(names)) != len(names):
        raise ValueError("Hay nombres de sesión repetidos; cada sesión debe ser única")
    return recordings


def save_dataset_csv(path, sessions, frame_rate, start_epoch=1.7931e9):
    """Escribe un CSV con el formato de la placa (solo las columnas usadas). Lo usa el simulador.

    sessions: lista de (nombre de sesión, escenario, CIR complejo [tramas, N_TAPS]) y, opcional,
    un cuarto elemento [tramas, 2] con (dist_cm, aoa_deg), NaN = sin estimación, y un quinto
    dict con "rx2" (complejo) y "presence" (0/1 por trama). El resto de columnas del formato
    de la placa se completa: valid (hay estimación), gap_ms (intervalo entre tramas) y pdoa.
    """
    re_cols, im_cols = cir_columns()
    parts = []
    rx2_re, rx2_im = cir_columns("rx2")
    pdoa_cols = [f"pdoa_{k}" for k in range(N_TAPS)]
    for name, scenario, cir, *extra in sessions:
        n = len(cir)
        link = extra[0] if extra else np.full((n, 2), np.nan)
        more = extra[1] if len(extra) > 1 else {}
        rx2 = more.get("rx2", cir)
        times = np.round(start_epoch + np.arange(n) / frame_rate, 3)
        head = pd.DataFrame({
            "session": name, "scenario": scenario, "idx": np.arange(n), "time_epoch": times,
            "presence": np.asarray(more.get("presence", np.zeros(n)), int),
            "valid": np.isfinite(link).all(axis=1).astype(int),
            "dist_cm": np.round(link[:, 0], 1), "aoa_deg": np.round(link[:, 1], 1),
            "gap_ms": np.round(np.r_[0.0, np.diff(times)] * 1000, 1)})
        pdoa = np.angle(cir * np.conj(rx2))
        parts.append(pd.concat([head,
                                pd.DataFrame(np.round(cir.real, 2), columns=re_cols),
                                pd.DataFrame(np.round(cir.imag, 2), columns=im_cols),
                                pd.DataFrame(np.round(rx2.real, 2), columns=rx2_re),
                                pd.DataFrame(np.round(rx2.imag, 2), columns=rx2_im),
                                pd.DataFrame(np.round(pdoa, 3), columns=pdoa_cols)], axis=1))
        start_epoch += n / frame_rate + 300
    pd.concat(parts, ignore_index=True).to_csv(path, index=False, float_format="%.3f")


def window_label(raw_labels):
    """(escenario, clase) de una ventana, o None si mezcla escenarios."""
    unique = set(raw_labels)
    unknown = unique - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"Escenarios desconocidos {sorted(unknown)}; usa uno de {sorted(LABEL_MAP)} "
                         "o añádelo a LABEL_MAP en config.py")
    if len(unique) != 1:
        return None
    raw = unique.pop()
    return raw, LABEL_MAP[raw]


def build_feature_table(data, window_s=WINDOW_S, hop_s=HOP_S):
    """Una fila por ventana, con las características y la sesión de la que sale."""
    rows = []
    candidate_windows = 0
    mixed_windows = 0
    for rec in load_recordings(data):
        rate = rec["frame_rate"]
        for start, end in sliding_windows(len(rec["cir"]), rate, window_s, hop_s):
            candidate_windows += 1
            label = window_label(rec["raw_labels"][start:end])
            if label is None:   # ventana a caballo entre dos escenarios
                mixed_windows += 1
                continue
            row = extract_features(rec["cir"][start:end], rate, rec["link"][start:end])
            row["session"] = rec["session"]
            row["raw_label"], row["label"] = label
            row["scenario"] = label[0]
            row["t_start_s"] = start / rate
            rows.append(row)
    if not rows:
        raise ValueError(f"No hay ventanas válidas en {data}: {candidate_windows} candidatas, "
                         f"{mixed_windows} descartadas por mezclar escenarios; "
                         f"se necesitan {window_s} s de tramas por ventana")
    return pd.DataFrame(rows)


def feature_columns(table):
    return [c for c in table.columns if c not in META_COLUMNS]


def session_hashes(data):
    """Huella del contenido de cada sesión, para detectar grabaciones repetidas aunque se renombren."""
    return {hashlib.sha256(np.ascontiguousarray(rec["cir"]).tobytes()).hexdigest()
            for rec in load_recordings(data)}


def dataset_fingerprint(data):
    """Huella del contenido de los CSV, no solo de sus nombres o tamaños."""
    digest = hashlib.sha256()
    for path in dataset_files(data):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
