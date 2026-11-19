"""Adaptador ML: reproduce un ml_dataset.csv trama a trama y traduce la salida al contrato de la UI.

Es el UNICO archivo de la interfaz que conoce el ML (uwb.realtime.PresenceDetector).
Para cambiar de modelo solo hay que cambiar la ruta del .joblib.
"""
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]      # carpeta "uwb ML", para poder importar el paquete uwb
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uwb.realtime import PresenceDetector       # noqa: E402


def _num(v):
    return None if v is None or not np.isfinite(v) else float(v)


async def events(cfg, csv_path, model_path, session=None, rate_hz=20.0):
    df = pd.read_csv(csv_path)
    if session is None:
        session = df["session"].iloc[0]
    df = df[df["session"] == session].sort_values("idx").reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"Sesión '{session}' no encontrada en {csv_path}")
    taps = sorted(int(c.split("_")[2]) for c in df.columns if c.startswith("rx1_i_"))
    frames = (df[[f"rx1_i_{k}" for k in taps]].to_numpy(float)
              + 1j * df[[f"rx1_q_{k}" for k in taps]].to_numpy(float))
    dist = df["dist_cm"].to_numpy(float) if "dist_cm" in df else np.full(len(df), np.nan)
    aoa = df["aoa_deg"].to_numpy(float) if "aoa_deg" in df else np.full(len(df), np.nan)
    present = df["presence"].to_numpy(int) if "presence" in df else np.zeros(len(df), int)

    detector = PresenceDetector(model_path, frame_rate=rate_hz)
    label, p = "warming_up", None
    while True:
        for i, frame in enumerate(frames):
            res = detector.push(frame, dist[i], aoa[i])
            if res is not None:
                label, p = res[0], _num(res[1])
            d, a = _num(dist[i]), _num(aoa[i])
            unlock = bool(label == cfg["positive_label"] and d is not None and d <= cfg["unlock_radius_m"] * 100)
            yield {"t": time.time(), "label": label, "p_positive": p, "dist_cm": d, "aoa_deg": a,
                   "board_present": bool(present[i]), "unlock": unlock, "source": "csv"}
            await asyncio.sleep(1 / rate_hz)
        detector = PresenceDetector(model_path, frame_rate=rate_hz)   # reinicia la ventana al repetir el bucle
        label, p = "warming_up", None
