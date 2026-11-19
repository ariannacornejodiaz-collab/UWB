"""Entrena y evalúa Gradient Boosting. Resultados en results/gb/.

    .venv/bin/python train_gb.py
    .venv/bin/python train_gb.py --data data/real --tune
"""
from uwb.training import run

if __name__ == "__main__":
    run("gb")
