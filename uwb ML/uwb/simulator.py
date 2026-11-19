"""Simulador de CIR de radar UWB con la forma de los datos de la placa.

Escribe un CSV con todas las columnas que exporta la placa, así el resto del código no
distingue entre datos simulados y reales. Sirve para probar el código, no para
estimar la precisión real: la atenuación de la pared y el eco de la persona son
inventados. El acoplamiento, el ruido y la cadencia se han aproximado a las
grabaciones de data/real.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import FRAME_RATE_HZ, N_TAPS, TAP_M, WAVELENGTH_M, ZERO_TAP
from .dataset import save_dataset_csv

PULSE_SIGMA_TAPS = 1.0     # pulso de 2 ns muestreado a 1 ns
COUPLING_AMP = 27000.0     # pico de acoplamiento medido en data/real
NOISE_SIGMA = 4.0          # por componente I/Q y por trama, medido en los taps lejanos
PERSON_AMP_1M = 400.0      # eco de una persona en movimiento a 1 m
BREATH_RCS_DB = -16.4      # pecho respirando frente a cuerpo en movimiento (nota de NXP)
DRIFT_STEP = 2e-4          # deriva del acoplamiento por trama (amplitud y fase)


# Dificultad de una sesión. "hard" da casos poco claros: eco de persona y respiración débiles,
# más ruido, pared fina (fuera se parece a dentro), quietos que se mueven poco, estimaciones de
# la placa ruidosas, con huecos y a veces falsas, y una sala vacía con algo que oscila lentamente.
EASY = dict(noise=1.0, person=1.0, breath=1.0, wall_db=0.0, speed=1.0, dist_sd=5.0,
            aoa_sd_in=3.0, aoa_sd_out=12.0, dropout=0.05, ghost=0.0, flutter=0.0)
HARD = dict(noise=1.6, person=0.5, breath=0.5, wall_db=9.0, speed=0.3, dist_sd=20.0,
            aoa_sd_in=15.0, aoa_sd_out=18.0, dropout=0.25, ghost=0.15, flutter=1.0)


@dataclass
class Room:
    room_id: str
    depth_m: float          # distancia de la placa a la pared del fondo
    wall_loss_db: float     # pérdida de ida y vuelta al atravesar la pared
    static: np.ndarray      # ecos fijos de paredes y muebles, complejo [N_TAPS]
    coupling: np.ndarray    # acoplamiento TX-RX, complejo [N_TAPS]


def _pulse(center_tap):
    """Pulso gaussiano centrado en un tap fraccionario. Acepta escalar o vector [tramas]."""
    taps = np.arange(N_TAPS)
    center = np.atleast_1d(center_tap)[:, None]
    return np.exp(-0.5 * ((taps[None, :] - center) / PULSE_SIGMA_TAPS) ** 2)


def _phase(rng, size=None):
    return np.exp(1j * rng.uniform(0, 2 * np.pi, size))


def make_room(rng, room_id):
    depth = rng.uniform(3.0, 6.0)
    taps = np.arange(N_TAPS)

    # Pico principal y los rebotes cercanos que se ven en los taps 8 y 12 de data/real
    tail = np.where(taps >= ZERO_TAP, np.exp(-(taps - ZERO_TAP) / 8.0), 0.0)
    coupling = (
        COUPLING_AMP * _pulse(ZERO_TAP)[0] * _phase(rng)
        + 0.6 * COUPLING_AMP * _pulse(ZERO_TAP + 3)[0] * _phase(rng)
        + 0.3 * COUPLING_AMP * _pulse(ZERO_TAP + 7)[0] * _phase(rng)
        + 0.05 * COUPLING_AMP * tail * _phase(rng, N_TAPS)
    )

    static = 3000.0 / depth**2 * _pulse(ZERO_TAP + depth / TAP_M)[0] * _phase(rng)
    for dist in rng.uniform(0.5, 12.0, size=6):
        amp = rng.uniform(100, 1500) / max(dist, 1.0)
        static = static + amp * _pulse(ZERO_TAP + dist / TAP_M)[0] * _phase(rng)

    return Room(room_id, depth, rng.uniform(6.0, 16.0), static, coupling)


def _walk(rng, n, frame_rate, lo, hi, speed_scale=1.0):
    """Trayectoria radial de alguien andando entre lo y hi metros."""
    dist = np.empty(n)
    pos = rng.uniform(lo, hi)
    speed = 0.0
    for i in range(n):
        if i % int(2 * frame_rate) == 0:
            speed = speed_scale * rng.choice([-1, 1]) * rng.uniform(0.3, 1.2)
        pos += speed / frame_rate
        if pos < lo or pos > hi:
            speed = -speed
            pos = min(max(pos, lo), hi)
        dist[i] = pos
    return dist


def _smooth_gain(rng, n, frame_rate):
    """Fluctuación lenta de la sección radar del cuerpo al moverse."""
    kernel = np.ones(int(frame_rate)) / int(frame_rate)
    return np.exp(0.5 * np.convolve(rng.normal(size=n), kernel, mode="same") * np.sqrt(frame_rate))


def _noise(rng, n):
    return NOISE_SIGMA * (rng.normal(size=(n, N_TAPS)) + 1j * rng.normal(size=(n, N_TAPS)))


def simulate_session(room, position, activity, duration_s, rng, frame_rate=FRAME_RATE_HZ, hard=False):
    """Devuelve (CIR complejo [tramas, N_TAPS], link [tramas, 2] con dist_cm y aoa_deg) de una grabación.

    link imita lo que estima la placa: sin persona no hay estimación (NaN); con persona, la
    distancia lleva ruido de ~5 cm y el ángulo es más estable dentro que fuera. Los valores
    son inventados, como el resto del simulador.

    position: "none" (sala vacía), "inside" u "outside". activity: "walking" o "still"
    (se ignora si none).
    """
    k = HARD if hard else EASY
    n = int(duration_s * frame_rate)
    t = np.arange(n) / frame_rate

    drift = (1 + np.cumsum(rng.normal(0, DRIFT_STEP, n))) * np.exp(
        1j * np.cumsum(rng.normal(0, DRIFT_STEP, n))
    )
    clean = room.coupling[None, :] * drift[:, None] + room.static[None, :]

    link = np.full((n, 2), np.nan)
    if position != "none":
        if position == "inside":
            lo, hi, wall = 0.6, room.depth_m - 0.3, 1.0
        else:
            lo, hi, wall = room.depth_m + 0.3, room.depth_m + 3.0, 10 ** ((k["wall_db"] - room.wall_loss_db) / 20)

        if activity == "walking":
            dist = _walk(rng, n, frame_rate, lo, hi, k["speed"])
            amp = k["person"] * PERSON_AMP_1M * _smooth_gain(rng, n, frame_rate)
        else:
            breath = k["breath"] * rng.uniform(2e-3, 6e-3) * np.sin(2 * np.pi * rng.uniform(0.2, 0.35) * t)
            sway = np.cumsum(rng.normal(0, 1e-4, n))
            dist = rng.uniform(lo, hi) + breath + sway
            amp = k["person"] * PERSON_AMP_1M * 10 ** (BREATH_RCS_DB / 20) * np.ones(n)

        # Eco directo más un rebote más débil y más largo (multicamino)
        for extra_m, gain in [(0.0, 1.0), (rng.uniform(0.5, 2.0), 0.3)]:
            path = dist + extra_m
            echo = wall * gain * amp / path**2 * np.exp(-1j * 4 * np.pi * path / WAVELENGTH_M)
            clean = clean + echo[:, None] * _pulse(ZERO_TAP + path / TAP_M)

        angle_sd = k["aoa_sd_in"] if position == "inside" else k["aoa_sd_out"]
        aoa = rng.uniform(-40, 40) + np.cumsum(rng.normal(0, 0.05, n)) + rng.normal(0, angle_sd, n)
        link = np.column_stack([100 * dist + rng.normal(0, k["dist_sd"], n), aoa])
        link[rng.random(n) < k["dropout"]] = np.nan   # la placa pierde alguna estimación

    else:
        if k["flutter"]:   # cortina o planta que oscila a frecuencia de respiración
            d = rng.uniform(1.0, 5.0)
            wobble = d + 4e-3 * np.sin(2 * np.pi * rng.uniform(0.2, 0.35) * t)
            echo = 0.3 * PERSON_AMP_1M / wobble**2 * np.exp(-1j * 4 * np.pi * wobble / WAVELENGTH_M)
            clean = clean + echo[:, None] * _pulse(ZERO_TAP + wobble / TAP_M)
        if k["ghost"]:   # estimaciones falsas de la placa sin nadie
            fake = rng.random(n) < k["ghost"]
            link[fake] = np.column_stack([rng.uniform(100, 600, fake.sum()), rng.uniform(-60, 60, fake.sum())])

    return (clean + k["noise"] * _noise(rng, n)).astype(np.complex64), link


def board_columns(cir, link, activity, rng):
    """Columnas de la placa que el modelo no usa: rx2, presence. Inventadas, como el resto.

    rx2 es rx1 con otro ruido y un desfase que depende del ángulo estimado. presence imita un
    detector de la placa que ve a quien se mueve pero a menudo no a quien está quieto.
    """
    n = len(cir)
    phase = np.where(np.isfinite(link[:, 1]), np.pi * np.sin(np.deg2rad(np.nan_to_num(link[:, 1]))), 0.3)
    rx2 = cir * np.exp(1j * phase)[:, None] + _noise(rng, n) * 0.5
    seen = np.isfinite(link[:, 1]) & (rng.random(n) < (0.95 if activity == "walking" else 0.2))
    return {"rx2": rx2.astype(np.complex64), "presence": seen.astype(int)}


# (dónde está la persona, qué hace, escenario que escribiría la placa)
SCENARIOS = [
    ("none", "none", "fuera_no_zona_o_nadie"),
    ("inside", "walking", "fuera_zona"),
    ("inside", "still", "fuera_zona"),
    ("outside", "walking", "dentro"),
    ("outside", "still", "dentro"),
]

ROOM_ID = "sala"
ROOM_SEED = 0   # la sala es siempre la misma; `seed` solo cambia las grabaciones
DATASET_FILE = "ml_dataset.csv"


def generate_dataset(out_dir, reps=12, duration_s=30.0, seed=0, name_prefix="", hard=False,
                     filename=DATASET_FILE):
    """Genera un único ml_dataset.csv, con una sesión por escenario y repetición, en la única sala."""
    if reps < 1 or not np.isfinite(duration_s) or duration_s < 6:
        raise ValueError("Se necesita al menos una repetición y 6 s por grabación")
    if name_prefix and not all(c.isalnum() or c in "_-" for c in name_prefix):
        raise ValueError("name_prefix solo admite letras, números, _ y -")
    room = make_room(np.random.default_rng(ROOM_SEED), ROOM_ID)
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = []
    for position, activity, scenario in SCENARIOS:
        for rep in range(reps):
            name = f"{name_prefix}{scenario}_{activity}_{rep}"
            cir, link = simulate_session(room, position, activity, duration_s, rng, hard=hard)
            sessions.append((name, scenario, cir, link, board_columns(cir, link, activity, rng)))
    save_dataset_csv(out_dir / filename, sessions, FRAME_RATE_HZ)
    return len(sessions)