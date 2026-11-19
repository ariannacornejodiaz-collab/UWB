"""Gradient Boosting: árboles pequeños entrenados uno tras otro, cada uno
corrigiendo los errores de los anteriores.

Se usa HistGradientBoostingClassifier de scikit-learn, que es rápido y equivalente
en la práctica a LightGBM. Suele ser el que más acierta con datos tabulares, pero
es más sensible a los hiperparámetros que Random Forest.
"""
from sklearn.ensemble import HistGradientBoostingClassifier

NAME = "gb"
TITLE = "Gradient Boosting"

GRID = {
    "learning_rate": [0.05, 0.1],
    "max_depth": [None, 4],
    "max_iter": [100, 300],
}


def build():
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=0,
    )
