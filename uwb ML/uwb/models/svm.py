"""SVM con kernel RBF: busca la frontera entre clases con el mayor margen posible.

Necesita las características en la misma escala, por eso va detrás de un
StandardScaler. No da importancia de características directamente.
"""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

NAME = "svm"
TITLE = "SVM (RBF)"

GRID = {
    "svc__C": [0.3, 1, 10],
    "svc__gamma": ["scale", 0.01, 0.1],
}


def build():
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", random_state=0),
    )
