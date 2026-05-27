"""Méthodes d'optimisation couplées au solveur FreeFEM++.

Les variables globales `history`, `best_J`, `best_x` sont volontairement
conservées (lisibilité du code "minimal version"). Toute fonction
appelant un optimiseur réinitialise l'état via `reset_optimization`.
"""

import time
from typing import Callable, Optional

import numpy as np
from scipy.optimize import basinhopping, differential_evolution, minimize
from tqdm import tqdm

from .freefem_interface import run_solver
from .utils import save_best_design

# État global (réinitialisé par reset_optimization)
history: list = []
iteration_counter: int = 0
best_J: float = -np.inf
best_x: Optional[np.ndarray] = None


def reset_optimization() -> None:
    global history, iteration_counter, best_J, best_x
    history = []
    iteration_counter = 0
    best_J = -np.inf
    best_x = None


def evaluate(x, mesh_size: int = 50) -> float:
    """Évalue J(x) via FreeFEM et alimente l'historique.

    Renvoie -J (les optimiseurs scipy minimisent ; on maximise J).
    """
    global iteration_counter, best_J, best_x, history
    start = time.time()
    J = run_solver(x, mesh_size=mesh_size, doplot=0)
    elapsed = time.time() - start

    history.append({
        "iteration": iteration_counter,
        "k1": float(x[0]), "k2": float(x[1]), "k3": float(x[2]),
        "k4": float(x[3]), "k5": float(x[4]), "Bi": float(x[5]),
        "J": J, "time": elapsed, "mesh_size": mesh_size,
    })

    if J > best_J:
        best_J = J
        best_x = np.array(x, dtype=float).copy()
        save_best_design(best_x, best_J)

    print(f"  Éval {iteration_counter:3d} | J = {J:.8f} | temps = {elapsed:.2f}s")
    iteration_counter += 1
    return -J


def run_differential_evolution(
    bounds,
    maxiter: int = 10,
    popsize: int = 5,
    mesh_size: int = 50,
    seed: Optional[int] = 42,
    progress_cb: Optional[Callable[[int, int], None]] = None,
):
    reset_optimization()
    pbar = tqdm(total=maxiter, desc="Differential Evolution", unit="gen")
    state = {"gen": 0}

    def callback(xk, conv):
        state["gen"] += 1
        pbar.update(1)
        if progress_cb is not None:
            progress_cb(state["gen"], maxiter)
        return False

    start = time.time()
    result = differential_evolution(
        lambda x: evaluate(x, mesh_size=mesh_size),
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        callback=callback,
        seed=seed,
        polish=False,
    )
    pbar.close()
    elapsed = time.time() - start
    return {
        "method": "Differential Evolution",
        "best_J": -result.fun,
        "best_x": result.x,
        "n_eval": len(history),
        "time": elapsed,
        "success": bool(result.success),
        "message": str(result.message),
        "history": list(history),
    }


def run_nelder_mead(
    bounds,
    x0=None,
    maxiter: int = 100,
    mesh_size: int = 50,
    progress_cb: Optional[Callable[[int, int], None]] = None,
):
    reset_optimization()
    if x0 is None:
        x0 = [(b[0] + b[1]) / 2 for b in bounds]
    pbar = tqdm(total=maxiter, desc="Nelder-Mead", unit="iter")
    state = {"it": 0}

    def callback(xk):
        state["it"] += 1
        pbar.update(1)
        if progress_cb is not None:
            progress_cb(state["it"], maxiter)
        return False

    start = time.time()
    result = minimize(
        lambda x: evaluate(x, mesh_size=mesh_size),
        x0,
        method="Nelder-Mead",
        bounds=bounds,
        callback=callback,
        options={"maxiter": maxiter, "disp": False},
    )
    pbar.close()
    elapsed = time.time() - start
    return {
        "method": "Nelder-Mead",
        "best_J": -result.fun,
        "best_x": result.x,
        "n_eval": len(history),
        "time": elapsed,
        "success": bool(result.success),
        "message": str(result.message),
        "history": list(history),
    }


def run_basinhopping(
    bounds,
    x0=None,
    niter: int = 100,
    mesh_size: int = 50,
    seed: Optional[int] = 42,
    progress_cb: Optional[Callable[[int, int], None]] = None,
):
    reset_optimization()
    if x0 is None:
        x0 = [(b[0] + b[1]) / 2 for b in bounds]
    pbar = tqdm(total=niter, desc="Basinhopping", unit="iter")
    state = {"it": 0}

    def callback(x, f, accepted):
        state["it"] += 1
        pbar.update(1)
        if progress_cb is not None:
            progress_cb(state["it"], niter)

    start = time.time()
    result = basinhopping(
        lambda x: evaluate(x, mesh_size=mesh_size),
        x0,
        niter=niter,
        minimizer_kwargs={"bounds": bounds, "method": "L-BFGS-B"},
        callback=callback,
        seed=seed,
    )
    pbar.close()
    elapsed = time.time() - start
    return {
        "method": "Basinhopping",
        "best_J": -result.fun,
        "best_x": result.x,
        "n_eval": len(history),
        "time": elapsed,
        "success": bool(result.lowest_optimization_result.success),
        "message": str(result.message),
        "history": list(history),
    }
