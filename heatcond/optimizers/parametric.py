"""Optimiseurs pour le modèle paramétrique (géométrie + matériau).

On applique **les trois mêmes méthodes que la Partie I** — Nelder-Mead,
Differential Evolution et l'hybride Adam->L-BFGS — au problème enrichi, via les
implémentations **SciPy** de :mod:`heatcond.optimizers.algorithms` et un objet
:class:`heatcond.objective.ParametricObjective` partagé qui journalise l'historique
complet (Q, J, coût, masse, F, temps par évaluation) et la courbe best-so-far.
Chaque ``run_*`` renvoie un dictionnaire normalisé pour la comparaison.

Les variables discrètes (matériaux) sont gérées **par relaxation + arrondi**
(l'arrondi a lieu dans l'objectif), comme la DE de la Partie I : on ne passe donc
pas ``integrality`` à SciPy.
"""

import time
from typing import Callable, Optional

import numpy as np

from . import algorithms as algo
from ..materials import bounds, integrality, unpack, N_FINS, T_LO, T_HI, L_LO, L_HI, BI_LO, BI_HI
from ..objective import ParametricObjective

_BND = bounds()
_LO = np.array([b[0] for b in _BND], dtype=float)
_HI = np.array([b[1] for b in _BND], dtype=float)


def _midpoint():
    return 0.5 * (_LO + _HI)


# ---------------------------------------------------------------------------
# Les trois méthodes de la Partie I, appliquées au modèle paramétrique
# ---------------------------------------------------------------------------
def run_nelder_mead(obj: ParametricObjective, x0=None, maxiter: int = 400, **_):
    """Nelder-Mead de SciPy (16D, bornes natives ; matériaux relâchés + arrondis)."""
    obj.reset()
    x0 = _midpoint() if x0 is None else np.asarray(x0, float)
    t0 = time.time()
    best_x = algo.nelder_mead(obj.F, x0, _BND, maxiter=maxiter)
    return _pack("Nelder-Mead", obj, best_x, time.time() - t0)


def run_differential_evolution(obj: ParametricObjective, popsize: int = 5,
                               maxiter: int = 8, seed: int = 42,
                               mutation=(0.5, 1.0), recombination: float = 0.7,
                               strategy: str = "best1bin",
                               progress_cb: Optional[Callable[[int, int], None]] = None,
                               **_):
    """Differential Evolution de SciPy. Réglages identiques à ceux de la Partie I."""
    obj.reset()
    t0 = time.time()
    best_x = algo.differential_evolution(obj.F, _BND, popsize=popsize, maxiter=maxiter,
                                       mutation=mutation, recombination=recombination,
                                       strategy=strategy, seed=seed, integrality=None,
                                       progress_cb=progress_cb)
    return _pack("Differential Evolution", obj, best_x, time.time() - t0)


def run_adam_then_lbfgs(obj: ParametricObjective, x0=None, n_adam: int = 30,
                        maxiter_lbfgs: int = 50, lr: float = 0.05, **_):
    """Hybride Adam->L-BFGS (Adam FD d'amorçage, puis L-BFGS-B de SciPy)."""
    obj.reset()
    x0 = _midpoint() if x0 is None else np.asarray(x0, float)
    t0 = time.time()
    best_x = algo.adam_then_lbfgs(obj.F, x0, _BND, n_adam=n_adam,
                                maxiter_lbfgs=maxiter_lbfgs, lr=lr)
    return _pack("Adam -> L-BFGS", obj, best_x, time.time() - t0)


def run_bayesian(obj: ParametricObjective, n_calls: int = 80, n_initial_points: int = 16,
                 seed: int = 42, progress_cb: Optional[Callable[[int, int], None]] = None, **_):
    """Optimisation bayésienne (GP + Expected Improvement, via scikit-optimize).

    Les matériaux sont traités nativement comme variables entières (dimensions
    Integer). Nécessite ``scikit-optimize`` (dépendance optionnelle).
    """
    obj.reset()
    t0 = time.time()
    best_x = algo.bayesian(obj.F, _BND, n_calls=n_calls, n_initial_points=n_initial_points,
                           seed=seed, integrality=integrality().tolist(), progress_cb=progress_cb)
    return _pack("Bayesian opt.", obj, best_x, time.time() - t0)


# Registre {nom: fonction} pour l'interface et les boucles d'étude.
METHODS = {
    "Nelder-Mead": run_nelder_mead,
    "Differential Evolution": run_differential_evolution,
    "Adam -> L-BFGS": run_adam_then_lbfgs,
    "Bayesian opt.": run_bayesian,
}


# ---------------------------------------------------------------------------
# Raffinement local multi-fidélité : matériaux figés, on polit (t, l, Bi)
# à maillage fin avec Nelder-Mead (SciPy).
# ---------------------------------------------------------------------------
def refine_local(best_x, lam_cost: float = 0.0, lam_mass: float = 0.0,
                 mass_budget: Optional[float] = None, mesh_size: int = 50,
                 maxiter: int = 120, solver: Optional[Callable] = None):
    """Affine la géométrie continue d'un design (matériaux fixés) sur maillage fin."""
    m_fixed, t0, l0, Bi0 = unpack(best_x)
    fine = ParametricObjective(lam_cost, lam_mass, mass_budget,
                               mesh_size=mesh_size, solver=solver)

    y0 = np.concatenate([t0, l0, [Bi0]])
    cont_bounds = [(T_LO, T_HI)] * N_FINS + [(L_LO, L_HI)] * N_FINS + [(BI_LO, BI_HI)]

    def embed(y):
        return np.concatenate([m_fixed.astype(float), y])

    t_start = time.time()
    best_y = algo.nelder_mead(lambda y: fine.F(embed(y)), y0, cont_bounds, maxiter=maxiter)
    out = _pack("Refine (Nelder-Mead, fin)", fine, embed(best_y), time.time() - t_start)
    out["mesh_size"] = mesh_size
    return out


def multifidelity(lam_cost: float = 0.0, lam_mass: float = 0.0,
                  mass_budget: Optional[float] = None,
                  coarse_mesh: int = 20, fine_mesh: int = 50,
                  maxiter: int = 20, popsize: int = 8, seed: int = 42,
                  verbose: bool = False,
                  progress_cb: Optional[Callable[[int, int], None]] = None,
                  solver: Optional[Callable] = None):
    """Pipeline recommandé : DE global (maillage grossier) -> raffinement fin.

    Renvoie (résultat_global, résultat_raffiné).
    """
    coarse = ParametricObjective(lam_cost, lam_mass, mass_budget,
                                 mesh_size=coarse_mesh, verbose=verbose,
                                 solver=solver)
    glob = run_differential_evolution(coarse, maxiter=maxiter, popsize=popsize,
                                      seed=seed, progress_cb=progress_cb)
    fine = refine_local(glob["best_x"], lam_cost, lam_mass, mass_budget,
                        mesh_size=fine_mesh, solver=solver)
    return glob, fine


# ---------------------------------------------------------------------------
def _pack(name, obj: ParametricObjective, best_x, elapsed):
    info = obj.best_info or {}
    return {
        "method": name,
        "best_F": obj.best,
        "best_x": np.asarray(best_x, dtype=float),
        "best_Q": info.get("Q"),
        "best_J": info.get("J"),
        "best_cost": info.get("cost"),
        "best_mass": info.get("mass"),
        "n_eval": obj.n,
        "time": elapsed,
        "curve": list(obj.curve),
        "history": list(obj.history),
    }
