"""Inferencia en tiempo real: recibe tramas de una en una y devuelve la clase."""
from __future__ import annotations

from collections import deque

import joblib
import numpy as np
import pandas as pd

from .config import FRAME_RATE_HZ, HOP_S, N_TAPS, POSITIVE, RX, WINDOW_S
from .features import extract_features
from .preprocess import window_parameters


class PresenceDetector:
    """Uso: detector.push(trama) con cada CIR nuevo; devuelve None o (clase, p_inside).

    La trama es el CIR complejo de N_TAPS taps de la antena RX1 (columnas rx1_i_k + 1j * rx1_q_k).
    La clase es "inside" (hay alguien dentro) u "outside" (no hay nadie dentro).
    """

    def __init__(self, model_path, frame_rate=FRAME_RATE_HZ):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.columns = bundle["features"]
        if bundle.get("rx", RX) != RX:
            raise ValueError(f"El modelo usa {bundle['rx']} y el código {RX}; vuelve a entrenar")
        window_s, hop_s = bundle.get("window_s", WINDOW_S), bundle.get("hop_s", HOP_S)
        self.frame_rate = frame_rate
        size, self.hop = window_parameters(frame_rate, window_s, hop_s)
        self.buffer = deque(maxlen=size)
        self.link = deque(maxlen=size)
        self.since_last = 0

    def push(self, frame, dist_cm=np.nan, aoa_deg=np.nan):
        """dist_cm y aoa_deg son los de la placa para esta trama (NaN si no hay estimación)."""
        frame = np.asarray(frame)
        if frame.shape != (N_TAPS,):
            raise ValueError(f"Cada trama debe tener {N_TAPS} taps complejos, no {frame.shape}")
        if not np.isfinite(frame).all():
            raise ValueError("La trama contiene valores no finitos")
        self.buffer.append(frame.copy())
        self.link.append((float(dist_cm), float(aoa_deg)))
        self.since_last += 1
        if len(self.buffer) < self.buffer.maxlen or self.since_last < self.hop:
            return None
        self.since_last = 0

        feats = extract_features(np.array(self.buffer), self.frame_rate, np.array(self.link))
        X = pd.DataFrame([feats])[self.columns]
        label = self.model.predict(X)[0]
        p_inside = float("nan")
        if hasattr(self.model, "predict_proba"):
            p_inside = float(self.model.predict_proba(X)[0, list(self.model.classes_).index(POSITIVE)])
        return str(label), p_inside
