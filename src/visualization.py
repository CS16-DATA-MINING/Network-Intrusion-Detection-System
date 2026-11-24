"""Figures statiques (matplotlib) pour les notebooks et le rapport PDF.

Thème clair (impression), palette catégorielle validée (ordre fixe), couleurs de
statut réservées : vert = normal, rouge = attaque.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src import config as C

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NORMAL, ATTACK = "#0ca30c", "#d03b3b"
SEQ_CMAP = "Blues"
DIV_CMAP = "RdBu_r"


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#0b0b0b", "axes.titleweight": "bold",
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": "#52514e", "ytick.color": "#52514e", "font.size": 10,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES), "legend.frameon": False,
        "font.family": "DejaVu Sans",
    })


def figure_dir(name: str):
    """reports/figures/eda pour l'EDA et le prétraitement (01_, 02_), evaluation sinon."""
    return C.EDA_FIG_DIR if name.startswith(("01_", "02_")) else C.EVAL_FIG_DIR


def save(fig, name: str) -> str:
    d = figure_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.png"
    fig.savefig(path)
    return str(path)


def class_colors(labels) -> list[str]:
    return [NORMAL if str(c) in ("Normal", "0") else ATTACK for c in labels]


def barh(ax, labels, values, color=SERIES[0], fmt="{:.3f}", annotate=True):
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, height=0.65)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    if annotate:
        for i, v in enumerate(values):
            ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8, color="#52514e")
    return ax


setup()
