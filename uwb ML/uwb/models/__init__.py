"""Un módulo por detector. Cada uno define NAME, TITLE, build() y GRID.

- build(): el estimador de scikit-learn sin entrenar, con los hiperparámetros por defecto.
- GRID: rejilla de hiperparámetros que se prueba con --tune.
"""
from . import gradient_boosting, random_forest, svm, threshold

MODULES = {m.NAME: m for m in [random_forest, gradient_boosting, svm, threshold]}
