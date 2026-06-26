"""Études de sensibilité.

- sensitivity_to_initial_point : variabilité du J final selon le point de départ
  (utile pour les optimiseurs locaux comme Nelder-Mead).
- parameter_sweep              : balayage de chaque paramètre indépendamment
  (one-at-a-time) autour d'un point de référence.
- compare_mesh_sizes           : J optimisé en fonction du maillage
  (mesh-sensitivity, Task 10 du sujet).
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from .freefem import run_solver
from .optimizers.simple import (
    reset_optimization,
    run_basinhopping,
    run_nelder_mead,
)


def sensitivity_to_initial_point(
    bounds,
    n_trials: int = 5,
    method: str = "nelder_mead",
    maxiter: int = 30,
    mesh_size: int = 50,
    seed: Optional[int] = 0,
):
    """Lance plusieurs optimisations locales depuis des x0 aléatoires."""
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_trials):
        x0 = rng.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        print(f"\nEssai {i + 1}/{n_trials} — x0 = {x0.round(4)}")
        if method == "nelder_mead":
            res = run_nelder_mead(bounds, x0=x0, maxiter=maxiter, mesh_size=mesh_size)
        elif method == "basinhopping":
            res = run_basinhopping(bounds, x0=x0, niter=maxiter // 2, mesh_size=mesh_size)
        else:
            raise ValueError(f"Méthode non supportée : {method}")
        results.append(res["best_J"])
        reset_optimization()
    arr = np.array(results)
    print(f"\n--- Résumé sensibilité ({method}) ---")
    print(f"Moyenne J = {arr.mean():.6f} ± {arr.std():.6f}")
    print(f"Min = {arr.min():.6f}, Max = {arr.max():.6f}")
    return results


def parameter_sweep(
    x_ref: Sequence[float],
    bounds,
    n_points: int = 11,
    mesh_size: int = 50,
) -> Dict[str, Dict[str, List[float]]]:
    """Balayage one-at-a-time : pour chaque paramètre i, on fait varier x[i]
    sur n_points dans bounds[i] en gardant les autres à x_ref.

    Renvoie : {"k1": {"values": [...], "J": [...]}, ...}
    """
    names = ["k1", "k2", "k3", "k4", "k5", "Bi"]
    out: Dict[str, Dict[str, List[float]]] = {}
    for i, name in enumerate(names):
        low, high = bounds[i]
        values = np.linspace(low, high, n_points).tolist()
        Js: List[float] = []
        for v in values:
            x = list(x_ref)
            x[i] = v
            J = run_solver(x, mesh_size=mesh_size, doplot=0)
            Js.append(J)
            print(f"  sweep {name} = {v:.4f} -> J = {J:.6f}")
        out[name] = {"values": values, "J": Js}
    return out


def compare_mesh_sizes(
    bounds,
    mesh_sizes: Sequence[int] = (25, 50, 100),
    method: str = "nelder_mead",
    maxiter: int = 30,
) -> Dict[int, float]:
    """Optimise pour différentes tailles de maillage et compare le J final."""
    results: Dict[int, float] = {}
    for ms in mesh_sizes:
        print(f"\n--- Maillage {ms}x{ms} ---")
        if method == "nelder_mead":
            res = run_nelder_mead(bounds, maxiter=maxiter, mesh_size=ms)
        elif method == "basinhopping":
            res = run_basinhopping(bounds, niter=maxiter // 2, mesh_size=ms)
        else:
            raise ValueError(f"Méthode non supportée : {method}")
        results[ms] = res["best_J"]
        reset_optimization()
    print("\n--- Comparaison des maillages ---")
    for ms, J in results.items():
        print(f"  {ms}x{ms} -> J = {J:.6f}")
    return results
