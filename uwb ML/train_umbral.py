"""Evalúa el detector por umbral, la referencia DSP. Resultados en results/umbral/.

    .venv/bin/python train_umbral.py
    .venv/bin/python train_umbral.py --data data/real
"""
from uwb.training import run

if __name__ == "__main__":
    run("umbral")
