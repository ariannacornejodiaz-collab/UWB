"""Entrena y evalúa SVM. Resultados en results/svm/.

    .venv/bin/python train_svm.py
    .venv/bin/python train_svm.py --data data/real --tune
"""
from uwb.training import run

if __name__ == "__main__":
    run("svm")
