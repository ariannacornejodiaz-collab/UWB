"""Detector por umbral: el algoritmo DSP de referencia, no es ML.

Mira una sola característica (por defecto, la variación máxima sobre el ruido) y dice "inside" si supera un umbral. El umbral se aprende con las
ventanas de entrenamiento. Si un modelo de ML no lo supera con claridad, no
merece la pena usarlo.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ..config import CLASSES, POSITIVE

NAME = "umbral"
TITLE = "Umbral (DSP)"

GRID = {"feature": ["all_peak_db", "all_mean_db", "all_slow_peak_db"]}


class ThresholdDetector(ClassifierMixin, BaseEstimator):
    def __init__(self, feature="all_peak_db"):
        self.feature = feature

    def fit(self, X, y):
        x = X[self.feature].to_numpy(float)
        positive = np.asarray(y) == POSITIVE
        if positive.all() or not positive.any():
            raise ValueError("El umbral necesita ventanas de las dos clases para entrenarse")
        # Prueba como umbral cada punto medio entre valores y se queda con el de mayor
        # exactitud equilibrada (media del acierto en cada clase)
        values = np.unique(x)
        candidates = (values[:-1] + values[1:]) / 2 if len(values) > 1 else values
        scores = [(np.mean(x[positive] > c) + np.mean(x[~positive] <= c)) / 2 for c in candidates]
        self.threshold_ = float(candidates[int(np.argmax(scores))])
        self.scale_ = float(np.std(x)) or 1.0
        self.classes_ = np.array(CLASSES)
        return self

    def predict_proba(self, X):
        # No es una probabilidad de verdad: solo distancia al umbral pasada por una sigmoide,
        # para poder calcular el AUC y dar una "confianza" en tiempo real
        x = X[self.feature].to_numpy(float)
        p_inside = 1 / (1 + np.exp(-(x - self.threshold_) / (0.1 * self.scale_)))
        proba = np.empty((len(x), len(self.classes_)))
        proba[:, list(self.classes_).index(POSITIVE)] = p_inside
        proba[:, [i for i, c in enumerate(self.classes_) if c != POSITIVE]] = (1 - p_inside)[:, None]
        return proba

    def predict(self, X):
        x = X[self.feature].to_numpy(float)
        other = [c for c in self.classes_ if c != POSITIVE][0]
        return np.where(x > self.threshold_, POSITIVE, other)


def build():
    return ThresholdDetector()
