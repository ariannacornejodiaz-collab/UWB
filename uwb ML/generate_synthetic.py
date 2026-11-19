"""Genera un dataset sintético, con el mismo formato CSV que la placa, para probar el pipeline.

    .venv/bin/python generate_synthetic.py --out data/synthetic
"""
import argparse

from uwb.simulator import generate_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--reps", type=int, default=12, help="repeticiones por escenario")
    parser.add_argument("--duration", type=float, default=30.0, help="segundos por sesión")
    parser.add_argument("--seed", type=int, default=0, help="semilla de las grabaciones (la sala es fija)")
    parser.add_argument("--name-prefix", default="", help="prefijo de los nombres de sesión")
    parser.add_argument("--hard", action="store_true", help="casos poco claros para poner a prueba los modelos")
    parser.add_argument("--file", default="ml_dataset.csv", help="nombre del CSV de salida")
    args = parser.parse_args()

    n = generate_dataset(args.out, args.reps, args.duration, args.seed, args.name_prefix, args.hard, args.file)
    print(f"{n} sesiones guardadas en {args.out}")
