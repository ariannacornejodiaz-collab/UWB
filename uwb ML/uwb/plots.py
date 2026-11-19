"""Gráficas que guardan los scripts de entrenamiento y comparación."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paleta categórica en orden fijo: cada detector conserva siempre su color
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DETECTOR_ORDER = ["rf", "gb", "svm", "umbral"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "axes.titlecolor": TEXT, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def detector_color(name):
    index = DETECTOR_ORDER.index(name) if name in DETECTOR_ORDER else len(DETECTOR_ORDER)
    return SERIES[index % len(SERIES)]


def confusion(matrix, classes, title, out):
    """Matriz de confusión con recuentos y porcentaje por fila (lo que había de verdad)."""
    share = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ink = "white" if share[i, j] > 0.6 else TEXT
            ax.text(j, i, f"{matrix[i, j]}\n{share[i, j]:.1%}", ha="center", va="center", color=ink)
    ax.set_xticks(range(len(classes)), [f"pred. {c}" for c in classes])
    ax.set_yticks(range(len(classes)), [f"real {c}" for c in classes])
    ax.set_title(title, fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def importance(ranking, title, out, top=15):
    """Barras horizontales con la caída de acierto al desordenar cada característica."""
    ranking = ranking.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(ranking) + 1.2))
    ax.barh(ranking.index, ranking.values, height=0.6, color=SERIES[0])
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.set_xlabel("Caída de exactitud al desordenarla")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def comparison(summary, metrics, labels, out):
    """Barras agrupadas: una por detector dentro de cada métrica."""
    names = list(summary.index)
    width = 0.8 / len(names)
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for k, name in enumerate(names):
        values = summary.loc[name, metrics].to_numpy(float)
        bars = ax.bar(x + (k - (len(names) - 1) / 2) * width, values, width * 0.92,
                      color=detector_color(name), label=summary.loc[name, "title"])
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in values], fontsize=7.5, color=TEXT_2, padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=len(names), loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
