"""Resume un CSV de la placa (formato ml_dataset.csv): sesiones, duración, cadencia, escenarios y pico.

    python inspect_csv.py data/real/ml_dataset.csv

Sirve para comprobar cada grabación nueva antes de usarla, y para confirmar
ZERO_TAP y FRAME_RATE_HZ de config.py.
"""
import argparse
import warnings
from collections import Counter

import numpy as np

from uwb.config import MIN_RANGE_M, RX, ZERO_TAP, m_to_tap, tap_to_m
from uwb.dataset import load_recordings
from uwb.preprocess import remove_clutter, sliding_windows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    for path in args.files:
        print(f"\n{path}")
        for rec in load_recordings(path):
            cir, rate = rec["cir"], rec["frame_rate"]
            peak = int(np.argmax(np.abs(cir).mean(axis=0)))

            # Dónde está lo que se mueve, tras restar el fondo en ventanas de 6 s
            windows = list(sliding_windows(len(cir), rate, 6, 6))
            if not windows:
                warnings.warn(f"{rec['session']}: menos de 6 s; se inspecciona la sesión completa")
                windows = [(0, len(cir))]
            power = np.mean([np.mean(np.abs(remove_clutter(cir[s:e])) ** 2, axis=0)
                             for s, e in windows], axis=0)
            first = m_to_tap(MIN_RANGE_M)
            moving = first + int(np.argmax(power[first:]))
            floor = np.median(power[first:])

            print(f"  sesión {rec['session']}: {len(cir)} tramas, {rec['elapsed_s']:.1f} s, "
                  f"{rate:.2f} tramas/s, {rec['lost_frames']} tramas perdidas")
            print(f"    escenarios: {dict(Counter(rec['raw_labels']))}")
            print(f"    {RX}: pico de acoplamiento en el tap {peak} "
                  f"(config.ZERO_TAP = {ZERO_TAP}){'' if peak == ZERO_TAP else '  <-- NO COINCIDE'}")
            print(f"    máxima variación a partir de {MIN_RANGE_M} m: tap {moving} = "
                  f"{tap_to_m(moving):.2f} m, {10 * np.log10(power[moving] / floor):.1f} dB sobre el ruido")