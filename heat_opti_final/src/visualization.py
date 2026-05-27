"""Visualisation : convergence, champs T, sweep, maillage.

Toutes les fonctions ont une variante "draw_*" qui peuple un Axes existant
(utile pour intégrer dans Tkinter via FigureCanvasTkAgg) et une variante
"plot_*" qui sauvegarde un PNG dans results/.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Convergence
# ===========================================================================
def _running_best(values: Sequence[float]) -> list:
    best = -float("inf")
    out = []
    for v in values:
        best = max(best, v)
        out.append(best)
    return out


def draw_convergence(ax, history) -> None:
    if not history:
        ax.text(0.5, 0.5, "Pas d'historique disponible.",
                ha="center", va="center", transform=ax.transAxes)
        return
    df = pd.DataFrame(history)
    ax.plot(_running_best(df["J"].tolist()), lw=2, color="#1f77b4")
    ax.set_xlabel("Évaluation")
    ax.set_ylabel("Meilleur J atteint")
    ax.set_title("Convergence de l'optimisation")
    ax.grid(True, alpha=0.4)


def plot_convergence(history, filename: str = "convergence.png",
                     show: bool = False) -> Path:
    _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_convergence(ax, history)
    path = RESULTS_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return path


def draw_convergence_comparison(ax, results_per_method: Dict[str, list]) -> None:
    for method_name, history in results_per_method.items():
        if not history:
            continue
        Js = [row["J"] for row in history]
        ax.plot(_running_best(Js), lw=2, label=method_name)
    ax.set_xlabel("Nombre d'évaluations")
    ax.set_ylabel("Meilleur J atteint")
    ax.set_title("Convergence comparée des optimiseurs")
    ax.grid(True, alpha=0.4)
    ax.legend()


def plot_convergence_comparison(
    results_per_method: Dict[str, list],
    filename: str = "convergence_comparison.png",
    show: bool = False,
) -> Path:
    _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_convergence_comparison(ax, results_per_method)
    plt.tight_layout()
    path = RESULTS_DIR / filename
    plt.savefig(path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return path


# ===========================================================================
# Maillage
# ===========================================================================
def draw_mesh(ax, vertices: np.ndarray, triangles: np.ndarray,
              edges: Optional[np.ndarray] = None,
              show_labels: bool = True) -> None:
    """Trace le maillage : triangles en gris + bords colorés par label."""
    tri = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles[:, :3])
    ax.triplot(tri, "-", lw=0.4, color="#888")
    ax.set_aspect("equal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Maillage — {len(vertices)} sommets, {len(triangles)} triangles")

    if edges is not None and show_labels:
        # Couleur par label de bord
        unique_labels = np.unique(edges[:, 2])
        cmap = plt.get_cmap("tab10")
        # Légende : bottom=1 (Base), right=2/top=3/left=4 (Fin)
        names = {1: "Base (T=1)", 2: "Fin droite", 3: "Fin haut", 4: "Fin gauche"}
        for k, lbl in enumerate(unique_labels):
            mask = edges[:, 2] == lbl
            color = cmap(k % 10)
            for v1, v2, _ in edges[mask]:
                ax.plot([vertices[v1, 0], vertices[v2, 0]],
                        [vertices[v1, 1], vertices[v2, 1]],
                        "-", color=color, lw=1.6)
            ax.plot([], [], "-", color=color, lw=2,
                    label=names.get(int(lbl), f"label {int(lbl)}"))
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


# ===========================================================================
# Champ T
# ===========================================================================
def draw_temperature(ax, x: np.ndarray, y: np.ndarray, T: np.ndarray,
                     triangles: Optional[np.ndarray] = None,
                     title: str = "Champ de température T",
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None,
                     cmap: str = "inferno"):
    if triangles is not None:
        tri = mtri.Triangulation(x, y, triangles[:, :3])
    else:
        tri = mtri.Triangulation(x, y)
    levels = np.linspace(vmin, vmax, 25) if (vmin is not None and vmax is not None) else 25
    tcf = ax.tricontourf(tri, T, levels=levels, cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    return tcf


def plot_temperature_comparison(
    initial_T_file: Path,
    optimized_T_file: Path,
    initial_label: str = "Design initial",
    optimized_label: str = "Design optimisé",
    filename: str = "T_comparison.png",
    show: bool = False,
) -> Path:
    _ensure_results_dir()
    from .freefem_interface import read_temperature_field

    x0, y0, T0 = read_temperature_field(initial_T_file)
    x1, y1, T1 = read_temperature_field(optimized_T_file)
    vmin = min(T0.min(), T1.min())
    vmax = max(T0.max(), T1.max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    draw_temperature(axes[0], x0, y0, T0, title=initial_label, vmin=vmin, vmax=vmax)
    tcf = draw_temperature(axes[1], x1, y1, T1, title=optimized_label, vmin=vmin, vmax=vmax)
    fig.colorbar(tcf, ax=axes, fraction=0.04, pad=0.04, label="T")
    fig.suptitle("Champs de température : initial vs optimisé")
    path = RESULTS_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


# ===========================================================================
# Sensibilité paramètre par paramètre
# ===========================================================================
def plot_parameter_sweep(
    sweep_results: Dict[str, Dict[str, list]],
    filename: str = "parameter_sweep.png",
    show: bool = False,
) -> Path:
    _ensure_results_dir()
    names = list(sweep_results.keys())
    n = len(names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for i, name in enumerate(names):
        ax = axes[i // cols][i % cols]
        d = sweep_results[name]
        ax.plot(d["values"], d["J"], "o-", lw=2)
        ax.set_xlabel(name)
        ax.set_ylabel("J")
        ax.set_title(f"Sensibilité de J à {name}")
        ax.grid(True, alpha=0.4)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Sensibilité paramètre par paramètre (one-at-a-time)")
    plt.tight_layout()
    path = RESULTS_DIR / filename
    plt.savefig(path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return path
