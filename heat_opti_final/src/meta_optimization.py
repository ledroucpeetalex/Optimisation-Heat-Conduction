"""Recherche par grille des hyperparamètres de Differential Evolution.

Conservé pour `run_sensitivity_study.py` ; utilise la nouvelle interface.
"""

import itertools
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .optimization import reset_optimization, run_differential_evolution


def grid_search_de(
    bounds,
    max_evals_per_config: int = 100,
    popsize_range: Sequence[int] = (4, 8, 12),
    mesh_size: int = 25,
) -> Tuple[Optional[Tuple[int]], List[Tuple[int, float, int, float]]]:
    """Balaye `popsize` pour DE et renvoie la meilleure configuration.

    Note : `mutation` et `recombination` ne sont pas exposés par
    `run_differential_evolution` (volontairement, version minimale).
    """
    best_J = -np.inf
    best_params: Optional[Tuple[int]] = None
    results: List[Tuple[int, float, int, float]] = []

    for popsize in popsize_range:
        print(f"\n--- DE popsize={popsize} (mesh_size={mesh_size}) ---")
        reset_optimization()
        maxiter = max(1, max_evals_per_config // popsize)
        res = run_differential_evolution(
            bounds, maxiter=maxiter, popsize=popsize, mesh_size=mesh_size,
        )
        J = res["best_J"]
        results.append((popsize, J, res["n_eval"], res["time"]))
        print(f"  J = {J:.6f}  ({res['n_eval']} évaluations, {res['time']:.1f}s)")
        if J > best_J:
            best_J = J
            best_params = (popsize,)

    print("\n=== Meilleurs hyperparamètres ===")
    if best_params is None:
        print("Aucun résultat valide.")
        return None, results
    print(f"popsize={best_params[0]}, J={best_J:.6f}")
    return best_params, results
