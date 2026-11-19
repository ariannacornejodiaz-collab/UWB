"""Fuente de simulación pura (sin ML): una persona entra en el campo, se acerca, se queda y se va."""
import asyncio
import random
import time

RATE_HZ = 10

# (duración s, dist inicial cm, dist final cm, ángulo inicial, ángulo final); dist None = nadie
SCRIPT = [
    (4, None, None, 0, 0),
    (8, 750, 300, -50, -20),    # se acerca por la izquierda
    (4, 300, 120, -20, 0),      # entra en el radio de desbloqueo
    (5, 120, 120, 0, 3),        # parada
    (5, 120, 400, 3, 40),       # se aleja
    (4, 400, 600, 40, 80),      # sale por el lateral (fuera de campo)
    (3, None, None, 0, 0),
    (5, 700, 100, 55, 30),
    (4, 100, 100, 30, 30),
]


def _decide(dist_cm, cfg):
    useful = dist_cm is not None and cfg["blind_zone_m"] * 100 <= dist_cm <= cfg["useful_max_m"] * 100
    label = cfg["positive_label"] if useful else "outside"
    p = 0.97 if useful else 0.03
    unlock = bool(useful and dist_cm <= cfg["unlock_radius_m"] * 100)
    return label, p, unlock


async def events(cfg):
    while True:
        for dur, d0, d1, a0, a1 in SCRIPT:
            n = int(dur * RATE_HZ)
            for i in range(n):
                f = i / max(n - 1, 1)
                if d0 is None:
                    dist = aoa = None
                else:
                    dist = round((d0 + (d1 - d0) * f) + random.gauss(0, 3))
                    aoa = round((a0 + (a1 - a0) * f) + random.gauss(0, 1.5))
                label, p, unlock = _decide(dist, cfg)
                yield {"t": time.time(), "label": label, "p_positive": p, "dist_cm": dist, "aoa_deg": aoa,
                       "board_present": dist is not None, "unlock": unlock, "source": "sim"}
                await asyncio.sleep(1 / RATE_HZ)
