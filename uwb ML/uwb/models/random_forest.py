"""Random Forest: muchos árboles de decisión, cada uno entrenado con una muestra
distinta de ventanas y de características, que votan la clase.

Casi no necesita ajuste y da la importancia de cada característica. Es la
referencia natural entre los modelos de ML.
"""
from sklearn.ensemble import RandomForestClassifier

NAME = "rf"
TITLE = "Random Forest"

GRID = {
    "n_estimators": [200, 500],
    "max_depth": [None, 8],
    "min_samples_leaf": [1, 5],
}


def build():
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",   # compensa si hay más ventanas de una clase
        n_jobs=-1,
        random_state=0,
    )
