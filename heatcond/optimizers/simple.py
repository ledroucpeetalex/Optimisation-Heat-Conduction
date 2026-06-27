"""Pilote d'optimisation du modèle simple (Partie I), couplé au solveur FreeFEM++.

Ce module est spécifique au problème HEAT-COND original : il maximise la
température moyenne ``J`` pour le design ``x = [k1..k5, Bi]``, journalise chaque
évaluation FreeFEM dans un historique global, et renvoie des dictionnaires de
résultats normalisés.

Les **algorithmes** eux-mêmes ne sont pas réimplémentés ici : ils proviennent de
la couche générique :mod:`heatcond.optimizers.algorithms` (enveloppes
``scipy.optimize``), partagée avec le modèle paramétrique. Ce module se contente
de fournir l'objectif (J via FreeFEM, avec journalisation) et d'emballer les
résultats.

Drapeau global ``VERBOSE`` : mettre ``simple.VERBOSE = False`` pour silencer les
prints par évaluation (utile dans un notebook).
"""

import time
from typing import Callable, Optional

import numpy as np
from scipy.optimize import basinhopping

from . import algorithms as algo
from ..freefem import run_solver
from ..utils import save_best_design

# Drapeau de verbosité — affecte uniquement le print par évaluation.
VERBOSE: bool = True

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
    """Évalue J(x) via FreeFEM et alimente l'historique. Renvoie -J (à minimiser)."""
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

    if VERBOSE:
        print(f"  Éval {iteration_counter:3d} | J = {J:.8f} | temps = {elapsed:.2f}s")
    iteration_counter += 1
    return -J


def _objective(mesh_size: int):
    """Renvoie un callable feval(x) -> obj à MAXIMISER (et qui journalise)."""
    return lambda x: -evaluate(x, mesh_size=mesh_size)


def _default_x0(bounds):
    return np.array([(b[0] + b[1]) / 2 for b in bounds], dtype=float)


def _result(method, elapsed, extra=None):
    out = {
        "method": method,
        "best_J": best_J,
        "best_x": best_x,
        "n_eval": len(history),
        "time": elapsed,
        "success": True,
        "history": list(history),
    }
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Méthodes principales : délèguent à algorithms (couche partagée)
# ---------------------------------------------------------------------------
def run_differential_evolution(bounds, maxiter: int = 10, popsize: int = 5,
                               mesh_size: int = 50, seed: Optional[int] = 42,
                               progress_cb: Optional[Callable[[int, int], None]] = None,
                               **scipy_kwargs):
    """Differential Evolution (SciPy). `scipy_kwargs` : mutation, recombination, strategy…"""
    reset_optimization()
    t0 = time.time()
    algo.differential_evolution(_objective(mesh_size), bounds, popsize=popsize,
                              maxiter=maxiter, seed=seed, progress_cb=progress_cb,
                              **scipy_kwargs)
    return _result("Differential Evolution", time.time() - t0)


def run_nelder_mead(bounds, x0=None, maxiter: int = 100, mesh_size: int = 50,
                    progress_cb: Optional[Callable[[int, int], None]] = None,
                    **_):
    """Nelder-Mead (SciPy)."""
    reset_optimization()
    if x0 is None:
        x0 = _default_x0(bounds)
    t0 = time.time()
    algo.nelder_mead(_objective(mesh_size), x0, bounds, maxiter=maxiter,
                   progress_cb=progress_cb)
    return _result("Nelder-Mead", time.time() - t0)


def run_lbfgs_b(bounds, x0=None, maxiter: int = 50, mesh_size: int = 50,
                progress_cb: Optional[Callable[[int, int], None]] = None, **_):
    """L-BFGS-B (SciPy), un seul point de départ."""
    reset_optimization()
    if x0 is None:
        x0 = _default_x0(bounds)
    t0 = time.time()
    algo.lbfgs_b(_objective(mesh_size), x0, bounds, maxiter=maxiter,
               progress_cb=progress_cb)
    return _result("L-BFGS-B", time.time() - t0, {"x0": list(np.asarray(x0))})


def run_adam(bounds, x0=None, n_iters: int = 50, lr: float = 0.05,
             mesh_size: int = 50,
             progress_cb: Optional[Callable[[int, int], None]] = None, **_):
    """Adam (gradient par différences finies)."""
    reset_optimization()
    if x0 is None:
        x0 = _default_x0(bounds)
    t0 = time.time()
    algo.adam(_objective(mesh_size), x0, bounds, n_iters=n_iters, lr=lr,
            progress_cb=progress_cb)
    return _result("Adam", time.time() - t0, {"x0": list(np.asarray(x0))})


def run_adam_then_lbfgs(bounds, x0=None, n_adam_iters: int = 30, lr: float = 0.05,
                        maxiter_lbfgs: int = 50, mesh_size: int = 50,
                        progress_cb: Optional[Callable[[int, int], None]] = None,
                        **_):
    """Hybride Adam->L-BFGS-B : amorçage robuste puis raffinement superlinéaire."""
    reset_optimization()
    if x0 is None:
        x0 = _default_x0(bounds)
    t0 = time.time()
    algo.adam_then_lbfgs(_objective(mesh_size), x0, bounds, n_adam=n_adam_iters,
                       maxiter_lbfgs=maxiter_lbfgs, lr=lr, progress_cb=progress_cb)
    return _result("Adam → L-BFGS", time.time() - t0, {"x0": list(np.asarray(x0))})


# ---------------------------------------------------------------------------
# Méthodes auxiliaires (conservées pour les études de sensibilité)
# ---------------------------------------------------------------------------
def run_multistart_lbfgs(bounds, n_starts: int = 10, mesh_size: int = 50,
                         seed: Optional[int] = 42, maxiter_per_start: int = 50,
                         progress_cb: Optional[Callable[[int, int], None]] = None,
                         **_):
    """Multi-start L-BFGS-B : N départs aléatoires (via algorithms), on garde le meilleur."""
    reset_optimization()
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    t0 = time.time()
    feval = _objective(mesh_size)
    for k in range(n_starts):
        algo.lbfgs_b(feval, rng.uniform(lo, hi), bounds, maxiter=maxiter_per_start)
        if progress_cb is not None:
            progress_cb(k + 1, n_starts)
    return _result("Multi-start L-BFGS-B", time.time() - t0)


def run_basinhopping(bounds, x0=None, niter: int = 100, mesh_size: int = 50,
                     seed: Optional[int] = 42,
                     progress_cb: Optional[Callable[[int, int], None]] = None,
                     **scipy_kwargs):
    """Basinhopping de SciPy (recuit + minimisations locales L-BFGS-B)."""
    reset_optimization()
    if x0 is None:
        x0 = _default_x0(bounds)
    state = {"it": 0}

    def _cb(x, f, accepted):
        state["it"] += 1
        if progress_cb is not None:
            progress_cb(state["it"], niter)

    minimizer_kwargs = scipy_kwargs.pop("minimizer_kwargs",
                                        {"bounds": bounds, "method": "L-BFGS-B"})
    t0 = time.time()
    basinhopping(lambda x: evaluate(x, mesh_size=mesh_size), np.asarray(x0, float),
                 niter=niter, minimizer_kwargs=minimizer_kwargs, callback=_cb,
                 seed=seed, **scipy_kwargs)
    return _result("Basinhopping", time.time() - t0)
