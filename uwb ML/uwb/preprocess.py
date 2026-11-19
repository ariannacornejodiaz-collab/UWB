"""Resta de fondo y troceado en ventanas."""
from __future__ import annotations

import numpy as np

from .config import ZERO_TAP


def remove_clutter(window, compensate_drift=True):
    """Quita lo que no cambia dentro de la ventana: acoplamiento, paredes y muebles.

    El acoplamiento deriva con la temperatura del chip. Con compensate_drift, la
    referencia se reescala en cada trama con la ganancia compleja medida en el pico
    de acoplamiento, donde el eco de una persona es despreciable.
    """
    ref = window.mean(axis=0)
    if not compensate_drift:
        return window - ref
    peak = slice(max(ZERO_TAP - 1, 0), ZERO_TAP + 2)
    gain = (window[:, peak] @ ref[peak].conj()) / (np.vdot(ref[peak], ref[peak]).real + 1e-12)
    return window - gain[:, None] * ref[None, :]


def window_parameters(frame_rate, window_s, hop_s):
    """Valida los tiempos y devuelve el tamaño y salto en tramas."""
    for name, value in [("frame_rate", frame_rate), ("window_s", window_s), ("hop_s", hop_s)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} debe ser finito y mayor que cero")
    size = int(round(window_s * frame_rate))
    if size < 2:
        raise ValueError("La ventana debe contener al menos 2 tramas")
    hop = max(1, int(round(hop_s * frame_rate)))
    return size, hop


def sliding_windows(n_frames, frame_rate, window_s, hop_s):
    """Genera los slices (inicio, fin) de cada ventana sobre n_frames tramas."""
    size, hop = window_parameters(frame_rate, window_s, hop_s)
    for start in range(0, n_frames - size + 1, hop):
        yield start, start + size
