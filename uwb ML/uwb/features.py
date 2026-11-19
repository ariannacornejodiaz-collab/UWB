"""Características por ventana a partir del CIR complejo.

Las potencias se expresan respecto al suelo de ruido. Las distancias son absolutas
(metros desde la placa): el proyecto usa una sola sala, así que no se necesita
conocer la pared del fondo; el modelo aprende qué distancias son dentro y fuera.
"""
from __future__ import annotations

import numpy as np

from .config import BREATH_BAND_HZ, MIN_RANGE_M, N_TAPS, m_to_tap, tap_to_m
from .preprocess import remove_clutter

EPS = 1e-12
BAND_TAPS = 10   # 1,5 m por tramo


def _db(x):
    return 10 * np.log10(x + EPS)


def _zone(maps, lo, hi, prefix):
    """Características de los taps [lo, hi). Devuelve también el tap de máxima variación."""
    power, diff, slow, fast = maps["power"], maps["diff"], maps["slow"], maps["fast"]
    peak = lo + int(np.argmax(power[lo:hi]))
    return {
        f"{prefix}_peak_db": float(_db(power[peak] / maps["floor"])),
        f"{prefix}_mean_db": float(_db(power[lo:hi].mean() / maps["floor"])),
        # Movimiento lento (respiración) y rápido (andar), cada uno sobre su propio ruido
        f"{prefix}_slow_peak_db": float(_db(slow[lo:hi].max() / maps["slow_floor"])),
        f"{prefix}_fast_peak_db": float(_db(fast[lo:hi].max() / maps["fast_floor"])),
        # ~2 si es ruido, mucho menor si el cambio es lento
        f"{prefix}_diff_ratio": float(diff[peak] / (power[peak] + EPS)),
        f"{prefix}_breath_frac": float(slow[peak] / (slow[peak] + fast[peak] + EPS)),
    }, peak


def link_features(link):
    """Estadísticas de dist_cm y aoa_deg (calculados por la placa) en la ventana.

    link: [tramas, 2] con (dist_cm, aoa_deg); NaN donde la placa no estimó nada.
    Sin ninguna estimación, las estadísticas valen 0 y link_frac vale 0.
    """
    link = np.zeros((0, 2)) if link is None else np.asarray(link, float)
    ok = np.isfinite(link).all(axis=1) if len(link) else np.zeros(0, bool)
    feats = {"link_frac": float(ok.mean()) if len(ok) else 0.0}
    dist, aoa = (link[ok, 0] / 100, link[ok, 1]) if ok.any() else (np.zeros(1), np.zeros(1))
    feats["dist_mean_m"] = float(dist.mean())
    feats["dist_std_m"] = float(dist.std())
    feats["dist_range_m"] = float(np.ptp(dist))
    feats["aoa_mean_deg"] = float(aoa.mean())
    feats["aoa_std_deg"] = float(aoa.std())
    return feats


def extract_features(window, frame_rate, link=None):
    """window: CIR complejo [tramas, N_TAPS] sin procesar. link: opcional [tramas, 2] (dist_cm, aoa_deg).

    Devuelve un dict de floats.
    """
    if not np.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("frame_rate debe ser finito y mayor que cero")
    window = np.asarray(window)
    if window.ndim != 2 or window.shape[1] != N_TAPS or len(window) < 2:
        raise ValueError(f"La ventana debe contener al menos 2 tramas de {N_TAPS} taps")
    if not np.isfinite(window).all():
        raise ValueError("La ventana contiene valores no finitos")
    x = remove_clutter(window)
    first = m_to_tap(MIN_RANGE_M)

    # Espectro a lo largo del tiempo de cada tap, separado en banda lenta y rápida
    spectrum = np.abs(np.fft.fft(x * np.hanning(len(x))[:, None], axis=0)) ** 2
    freqs = np.abs(np.fft.fftfreq(len(x), 1 / frame_rate))
    slow_band = (freqs >= BREATH_BAND_HZ[0]) & (freqs <= BREATH_BAND_HZ[1])
    maps = {
        "power": np.mean(np.abs(x) ** 2, axis=0),                 # variación temporal por tap
        "diff": np.mean(np.abs(np.diff(x, axis=0)) ** 2, axis=0),  # cambio entre tramas seguidas
        "slow": spectrum[slow_band].sum(axis=0),
        "fast": spectrum[freqs > BREATH_BAND_HZ[1]].sum(axis=0),
    }
    maps["floor"] = float(np.median(maps["power"][first:])) + EPS
    maps["slow_floor"] = float(np.median(maps["slow"][first:])) + EPS
    maps["fast_floor"] = float(np.median(maps["fast"][first:])) + EPS
    power, floor = maps["power"], maps["floor"]

    feats, peak = _zone(maps, first, N_TAPS, "all")

    # Variación por tramos de distancia absoluta a la placa: con una sola sala, el modelo
    # aprende qué tramos corresponden a dentro y cuáles a fuera
    for k, lo in enumerate(range(first, N_TAPS, BAND_TAPS)):
        hi = min(lo + BAND_TAPS, N_TAPS)
        feats[f"band{k}_db"] = float(_db(power[lo:hi].mean() / floor))
        feats[f"band{k}_slow_db"] = float(_db(maps["slow"][lo:hi].mean() / maps["slow_floor"]))

    feats["peak_m"] = float(tap_to_m(peak))
    excess = np.clip(power[first:] - floor, 0, None)
    dist = tap_to_m(np.arange(first, N_TAPS))
    weight = excess.sum() + EPS
    centroid = float((excess * dist).sum() / weight)
    feats["centroid_m"] = centroid
    feats["spread_m"] = float(np.sqrt((excess * (dist - centroid) ** 2).sum() / weight))
    feats.update(link_features(link))
    return feats
