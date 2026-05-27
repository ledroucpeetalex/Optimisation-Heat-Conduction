"""Utilitaires : sauvegarde du meilleur design + historique CSV.

Plus de `best_params.txt` : le meilleur design est relu à partir du
CSV `results/optimization_history.csv`.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_best_design(best_x, best_J: float) -> None:
    """Sauvegarde une fiche lisible du meilleur design."""
    _ensure_results_dir()
    path = RESULTS_DIR / "best_design.txt"
    with open(path, "w") as f:
        f.write("===== BEST DESIGN =====\n\n")
        for i, name in enumerate(["k1", "k2", "k3", "k4", "k5", "Bi"]):
            f.write(f"{name} = {best_x[i]}\n")
        f.write(f"\nBest objective J = {best_J}\n")


def save_history(history, filename: str = "optimization_history.csv") -> Path:
    """Sauvegarde l'historique d'évaluations en CSV."""
    _ensure_results_dir()
    df = pd.DataFrame(history)
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    return path


def load_best_design():
    """Récupère le meilleur design depuis l'historique CSV.

    Renvoie une liste [k1, k2, k3, k4, k5, Bi] ou None si introuvable.
    """
    csv_path = RESULTS_DIR / "optimization_history.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    best_row = df.loc[df["J"].idxmax()]
    return [float(best_row[k]) for k in ["k1", "k2", "k3", "k4", "k5", "Bi"]]
