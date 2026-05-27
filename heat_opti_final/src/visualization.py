"""Visualisation : convergence, champs T, sweep, maillage.

Conventions :

- `draw_*` peuple un Axes existant (utilisable côté Tkinter via
  FigureCanvasTkAgg, ou côté notebook via pyplot).
- `plot_*` sauvegarde un PNG dans `results/` SANS toucher pyplot
  (utilise Figure + FigureCanvasAgg) — donc thread-safe : ces fonctions
  peuvent être appelées depuis le worker thread de l'app sans crasher
  TkAgg côté main thread.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: Figure, path: Path, dpi: int = 200,
          bbox_inches: str = "tight") -> None:
    """Sauvegarde une Figure via le backend Agg (pas de pyplot)."""
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(path, dpi=dpi, bbox_inches=bbox_inches)


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


def plot_convergence(history, filename: str = "convergence.png") -> Path:
    _ensure_results_dir()
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    draw_convergence(ax, history)
    fig.tight_layout()
    path = RESULTS_DIR / filename
    _save(fig, path)
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
) -> Path:
    _ensure_results_dir()
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    draw_convergence_comparison(ax, results_per_method)
    fig.tight_layout()
    path = RESULTS_DIR / filename
    _save(fig, path)
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
        # Convention (cf. mesh.edp) :
        #   1 = base   (Dirichlet T = 1)
        #   2 = TOUT le reste (Robin Bi*T + k*dT/dn = 0)
        names = {
            1: "Base (T = 1, Dirichlet)",
            2: "Surface fin (Robin)",
        }
        colors = {1: "#c0392b", 2: "#2980b9"}
        for lbl in np.unique(edges[:, 2]):
            mask = edges[:, 2] == lbl
            color = colors.get(int(lbl), "#7f8c8d")
            for v1, v2, _ in edges[mask]:
                ax.plot([vertices[v1, 0], vertices[v2, 0]],
                        [vertices[v1, 1], vertices[v2, 1]],
                        "-", color=color, lw=1.8)
            ax.plot([], [], "-", color=color, lw=2.5,
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
    """Trace tricontourf de T.

    Si `triangles` est fourni (typiquement issu de read_freefem_mesh), la
    triangulation native est utilisée — les vides entre ailettes restent
    vides. Sinon, fallback sur Delaunay sur (x, y) qui remplit l'enveloppe
    convexe (à éviter pour cette géométrie non convexe).
    """
    if triangles is not None:
        tri = mtri.Triangulation(x, y, triangles[:, :3])
    else:
        tri = mtri.Triangulation(x, y)
    if vmin is not None and vmax is not None:
        levels = np.linspace(vmin, vmax, 25)
    else:
        levels = 25
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
    mesh_path: Optional[Path] = None,
    initial_label: str = "Design initial",
    optimized_label: str = "Design optimisé",
    filename: str = "T_comparison.png",
) -> Path:
    """Sauvegarde la comparaison T initial / T optimisé.

    `mesh_path` doit pointer vers le .msh utilisé pour générer les .dat
    (sinon on retombe sur Delaunay -> enveloppe convexe).
    """
    _ensure_results_dir()
    from .freefem_interface import read_freefem_mesh, read_temperature_field

    x0, y0, T0 = read_temperature_field(initial_T_file)
    x1, y1, T1 = read_temperature_field(optimized_T_file)
    vmin = min(T0.min(), T1.min())
    vmax = max(T0.max(), T1.max())

    tri = None
    if mesh_path is not None:
        try:
            _, triangles, _ = read_freefem_mesh(mesh_path)
            if (len(triangles) and triangles[:, :3].max() < len(x0)
                    and triangles[:, :3].max() < len(x1)):
                tri = triangles
        except Exception:
            tri = None

    fig = Figure(figsize=(13, 6))
    ax1 = fig.add_subplot(121)
    draw_temperature(ax1, x0, y0, T0, triangles=tri,
                     title=initial_label, vmin=vmin, vmax=vmax)
    ax2 = fig.add_subplot(122)
    tcf = draw_temperature(ax2, x1, y1, T1, triangles=tri,
                           title=optimized_label, vmin=vmin, vmax=vmax)
    fig.colorbar(tcf, ax=[ax1, ax2], fraction=0.04, pad=0.04, label="T")
    fig.suptitle("Champs de température : initial vs optimisé")
    path = RESULTS_DIR / filename
    _save(fig, path)
    return path


# ===========================================================================
# Sensibilité paramètre par paramètre
# ===========================================================================
def plot_parameter_sweep(
    sweep_results: Dict[str, Dict[str, list]],
    filename: str = "parameter_sweep.png",
) -> Path:
    _ensure_results_dir()
    names = list(sweep_results.keys())
    n = len(names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig = Figure(figsize=(5 * cols, 4 * rows))
    for i, name in enumerate(names):
        ax = fig.add_subplot(rows, cols, i + 1)
        d = sweep_results[name]
        ax.plot(d["values"], d["J"], "o-", lw=2)
        ax.set_xlabel(name)
        ax.set_ylabel("J")
        ax.set_title(f"Sensibilité de J à {name}")
        ax.grid(True, alpha=0.4)
    fig.suptitle("Sensibilité paramètre par paramètre (one-at-a-time)")
    fig.tight_layout()
    path = RESULTS_DIR / filename
    _save(fig, path)
    return path
