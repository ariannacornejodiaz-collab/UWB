"""Mapa distancia-tiempo de una grabación tras restar el fondo.

    .venv/bin/python plot_session.py data/real/ml_dataset.csv [--session NOMBRE]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uwb.config import MIN_RANGE_M, N_TAPS, m_to_tap, tap_to_m
from uwb.dataset import load_recordings
from uwb.preprocess import remove_clutter, sliding_windows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="CSV de la placa")
    parser.add_argument("--session", help="sesión a pintar (por defecto, la primera)")
    parser.add_argument("--out", help="PNG de salida (por defecto, junto al CSV)")
    parser.add_argument("--window", type=float, default=2.0, help="segundos por fila del mapa")
    parser.add_argument("--max-range", type=float, default=10.0, help="distancia máxima a pintar (m)")
    args = parser.parse_args()

    recordings = load_recordings(args.data)
    data = next((r for r in recordings if r["session"] == args.session), None) if args.session else recordings[0]
    if data is None:
        raise SystemExit(f"No hay sesión {args.session}; hay: {[r['session'] for r in recordings]}")
    cir, rate = data["cir"], data["frame_rate"]
    rows = [np.mean(np.abs(remove_clutter(cir[s:e])) ** 2, axis=0)
            for s, e in sliding_windows(len(cir), rate, args.window, 0.5)]
    if not rows:
        raise SystemExit(f"No hay ventanas de {args.window} s en {data['session']}; "
                         "usa --window con una duración menor.")
    power = np.array(rows)
    first = m_to_tap(MIN_RANGE_M)
    floor = np.median(power[:, first:])
    image = 10 * np.log10(power / floor + 1e-12)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    extent = [tap_to_m(-0.5), tap_to_m(N_TAPS - 0.5), 0, len(cir) / rate]
    mesh = ax.imshow(image, aspect="auto", origin="lower", extent=extent, vmin=0, vmax=30, cmap="viridis")
    ax.set_xlim(extent[0], min(args.max_range, extent[1]))
    ax.set_xlabel("Distancia a la placa (m)")
    ax.set_ylabel("Tiempo (s)")
    labels = "/".join(sorted(set(data["raw_labels"])))
    ax.set_title(f"{data['session']} — {labels}")
    fig.colorbar(mesh, label="Variación sobre el ruido (dB)")
    fig.tight_layout()

    out = args.out or str(Path(args.data).with_name(f"{data['session']}.png"))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(out)
