"""Entrena y evalúa Random Forest. Resultados en results/rf/.

    .venv/bin/python train_rf.py
    .venv/bin/python train_rf.py --data data/real --tune
"""
from uwb.training import run

if __name__ == "__main__":
    run("rf")
