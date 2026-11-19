"""Reproduce un CSV de la placa trama a trama con PresenceDetector, como haría la interfaz.

    .venv/bin/python demo_realtime.py results/rf/model.joblib data/real/ml_dataset.csv [--session NOMBRE]

Imprime una decisión por segundo, tras los primeros 6 s de ventana.
"""
import argparse
from collections import Counter

from uwb.dataset import load_recordings
from uwb.realtime import PresenceDetector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("data", help="CSV de la placa")
    parser.add_argument("--session", help="sesión a reproducir (por defecto, la primera)")
    args = parser.parse_args()

    recordings = load_recordings(args.data)
    data = next((r for r in recordings if r["session"] == args.session), None) if args.session else recordings[0]
    if data is None:
        raise SystemExit(f"No hay sesión {args.session}; hay: {[r['session'] for r in recordings]}")
    detector = PresenceDetector(args.model, frame_rate=data["frame_rate"])
    decisions = Counter()
    for i, frame in enumerate(data["cir"]):
        result = detector.push(frame, *data["link"][i])
        if result is not None:
            label, p_inside = result
            decisions[label] += 1
            print(f"t={(i + 1) / data['frame_rate']:6.1f} s  {label:<8} p(inside)={p_inside:.2f}")
    print(f"\nresumen: {dict(decisions)} | escenario del CSV: {sorted(set(data['raw_labels']))}")
