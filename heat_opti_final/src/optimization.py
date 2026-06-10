"""Méthodes d'optimisation couplées au solveur FreeFEM++.

Drapeau global `VERBOSE` :
    Mettre `optimization.VERBOSE = False` pour silencer les prints par
    évaluation (utile dans un notebook).
"""

import time
from typing import Callable, Optional

import numpy as np
from scipy.optimize import basinhopping, differential_evolution, minimize
from tqdm import tqdm

from .freefem_interface import run_solver
from .utils import save_best_design

# Drapeau de verbosité — affecte uniquement le print par évaluation.
VERBOSE: bool = True

# État global (réinitialisé par reset_optimization)
history: list = []
iteration_counter: int = 0
best_J: float = -np.inf
best_x: Optional[np.ndarray] = None
best_obj: float = -np.inf

# Poids du coût matériau pour l'objectif scalarisé F = J - COST_LAMBDA * mean(k_i).
# 0.0 = objectif J pur (comportement par défaut). Voir l'extension front de Pareto.
COST_LAMBDA: float = 0.0


def reset_optimization() -> None:
    global history, iteration_counter, best_J, best_x, best_obj
    history = []
    iteration_counter = 0
    best_J = -np.inf
    best_x = None
    best_obj = -np.inf


def evaluate(x, mesh_size: int = 50) -> float:
    """Évalue J(x) via FreeFEM et alimente l'historique.

    Renvoie -J (les optimiseurs scipy minimisent ; on maximise J).
    """
    global iteration_counter, best_J, best_x, history, best_obj
    start = time.time()
    J = run_solver(x, mesh_size=mesh_size, doplot=0)
    elapsed = time.time() - start

    history.append({
        "iteration": iteration_counter,
        "k1": float(x[0]), "k2": float(x[1]), "k3": float(x[2]),
        "k4": float(x[3]), "k5": float(x[4]), "Bi": float(x[5]),
        "J": J, "time": elapsed, "mesh_size": mesh_size,
    })

    # Objectif scalarisé : J pénalisé par le coût matériau moyen.
    # COST_LAMBDA = 0 -> objectif J pur (best_obj == best_J, comportement initial).
    obj = J - COST_LAMBDA * float(np.mean(np.asarray(x[:5], dtype=float)))
    if obj > best_obj:
        best_obj = obj
        best_J = J
        best_x = np.array(x, dtype=float).copy()
        save_best_design(best_x, best_J)

    if VERBOSE:
        print(f"  Éval {iteration_counter:3d} | J = {J:.8f} | temps = {elapsed:.2f}s")
    iteration_counter += 1
    return -obj


def run_differential_evolution(
    bounds,
    maxiter: int = 10,
    popsize: int = 5,
    mesh_size: int = 50,
    seed: Optional[int] = 42,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    **scipy_kwargs,
):
    """Wrapper scipy.optimize.differential_evolution avec historique.

    `**scipy_kwargs` accepte tous les paramètres natifs de scipy :
    `mutation`, `recombination`, `strategy`, `tol`, `init`, etc.
    """
    reset_optimization()
    pbar = tqdm(total=maxiter, desc="Differential Evolution", unit="gen",
                disable=not VERBOSE)
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
        **scipy_kwargs,
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
    **scipy_kwargs,
):
    """Wrapper scipy.optimize.minimize(method='Nelder-Mead') avec historique."""
    reset_optimization()
    if x0 is None:
        x0 = [(b[0] + b[1]) / 2 for b in bounds]
    pbar = tqdm(total=maxiter, desc="Nelder-Mead", unit="iter",
                disable=not VERBOSE)
    state = {"it": 0}

    def callback(xk):
        state["it"] += 1
        pbar.update(1)
        if progress_cb is not None:
            progress_cb(state["it"], maxiter)
        return False

    options = scipy_kwargs.pop("options", {})
    options.setdefault("maxiter", maxiter)
    options.setdefault("disp", False)

    start = time.time()
    result = minimize(
        lambda x: evaluate(x, mesh_size=mesh_size),
        x0,
        method="Nelder-Mead",
        bounds=bounds,
        callback=callback,
        options=options,
        **scipy_kwargs,
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
    **scipy_kwargs,
):
    """Wrapper scipy.optimize.basinhopping avec historique.

    `**scipy_kwargs` accepte les paramètres natifs : `stepsize`, `T`,
    `disp`, etc.
    """
    reset_optimization()
    if x0 is None:
        x0 = [(b[0] + b[1]) / 2 for b in bounds]
    pbar = tqdm(total=niter, desc="Basinhopping", unit="iter",
                disable=not VERBOSE)
    state = {"it": 0}

    def callback(x, f, accepted):
        state["it"] += 1
        pbar.update(1)
        if progress_cb is not None:
            progress_cb(state["it"], niter)

    # Permettre à l'utilisateur de surcharger minimizer_kwargs sans collision
    user_mkw = scipy_kwargs.pop("minimizer_kwargs", None)
    if user_mkw is None:
        user_mkw = {"bounds": bounds, "method": "L-BFGS-B"}

    start = time.time()
    result = basinhopping(
        lambda x: evaluate(x, mesh_size=mesh_size),
        x0,
        niter=niter,
        minimizer_kwargs=user_mkw,
        callback=callback,
        seed=seed,
        **scipy_kwargs,
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


def run_multistart_lbfgs(
    bounds,
    n_starts: int = 10,
    mesh_size: int = 50,
    seed: Optional[int] = 42,
    maxiter_per_start: int = 50,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    **scipy_kwargs,
):
    """Multi-start L-BFGS-B : N points de départ aléatoires, on garde le meilleur.

    Approche classique pour fonctions lisses bornées : un local quasi-Newton
    rapide × quelques restarts uniformes bat souvent les méthodes globales
    populationnelles (DE) en nombre d'évaluations pour la même qualité de J*.

    Le gradient n'étant pas fourni (`jac=None`), scipy l'estime par différences
    finies — ce qui coûte 12 évaluations supplémentaires par estimation de
    gradient en dimension 6.
    """
    reset_optimization()
    rng = np.random.default_rng(seed)
    pbar = tqdm(total=n_starts, desc="Multi-start L-BFGS-B", unit="start",
                disable=not VERBOSE)

    best_J_global = -np.inf
    best_x_global = None
    per_start = []

    start_t = time.time()
    for k in range(n_starts):
        x0 = rng.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        n_eval_before = len(history)
        result = minimize(
            lambda x: evaluate(x, mesh_size=mesh_size),
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter_per_start, "disp": False},
            **scipy_kwargs,
        )
        local_J = -result.fun
        per_start.append({
            "k": k,
            "x0": list(x0),
            "x_local": list(result.x),
            "J_local": local_J,
            "n_eval_local": len(history) - n_eval_before,
            "success": bool(result.success),
        })
        if local_J > best_J_global:
            best_J_global = local_J
            best_x_global = np.array(result.x, dtype=float).copy()
        pbar.update(1)
        if progress_cb is not None:
            progress_cb(k + 1, n_starts)

    pbar.close()
    elapsed = time.time() - start_t
    return {
        "method": "Multi-start L-BFGS-B",
        "best_J": best_J_global,
        "best_x": best_x_global,
        "n_eval": len(history),
        "time": elapsed,
        "success": True,
        "message": f"{n_starts} starts completed",
        "history": list(history),
        "per_start": per_start,
    }


# ===========================================================================
# L-BFGS-B simple (un seul start), Adam, et hybride Adam → L-BFGS-B
# ===========================================================================
def _numerical_gradient(f, x, h: float = 1e-3):
    """Gradient par différences finies centrées. Coût : 2*n appels à f."""
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        x_p = x.copy(); x_p[i] += h
        x_m = x.copy(); x_m[i] -= h
        grad[i] = (f(x_p) - f(x_m)) / (2.0 * h)
    return grad


def run_lbfgs_b(
    bounds,
    x0=None,
    maxiter: int = 50,
    mesh_size: int = 50,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    **scipy_kwargs,
):
    """L-BFGS-B simple (un seul point de départ).

    Pour le multi-start, utiliser run_multistart_lbfgs() à la place.
    """
    reset_optimization()
    if x0 is None:
        x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])
    else:
        x0 = np.asarray(x0, dtype=float)

    pbar = tqdm(total=maxiter, desc="L-BFGS-B", unit="iter", disable=not VERBOSE)
    state = {"it": 0}

    def cb(xk):
        state["it"] += 1
        pbar.update(1)
        if progress_cb is not None:
            progress_cb(state["it"], maxiter)
        return False

    options = scipy_kwargs.pop("options", {})
    options.setdefault("maxiter", maxiter)
    options.setdefault("disp", False)

    start_t = time.time()
    result = minimize(
        lambda xx: evaluate(xx, mesh_size=mesh_size),
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        callback=cb,
        options=options,
        **scipy_kwargs,
    )
    pbar.close()
    elapsed = time.time() - start_t
    return {
        "method": "L-BFGS-B",
        "best_J": -result.fun,
        "best_x": result.x,
        "x0": list(x0),
        "n_eval": len(history),
        "time": elapsed,
        "success": bool(result.success),
        "message": str(result.message),
        "history": list(history),
    }


def run_adam(
    bounds,
    x0=None,
    n_iters: int = 50,
    lr: float = 0.05,
    lr_decay: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    h_grad: float = 1e-3,
    mesh_size: int = 50,
    progress_cb: Optional[Callable[[int, int], None]] = None,
):
    """Adam optimizer (Kingma & Ba, 2014) avec gradient par différences finies.

    Adam maintient deux moments exponentiels du gradient (m, v) et adapte
    le pas par coordonnée. Robuste aux mauvais points de départ.

    Coût par itération : 2 * dim = 12 évaluations FreeFEM en 6D (FD centré).

    Si lr_decay < 1.0, le pas est multiplié par lr_decay à chaque itération.
    Adam ascend J (équivalent à minimiser -J).
    """
    reset_optimization()
    if x0 is None:
        x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])
    else:
        x0 = np.asarray(x0, dtype=float)

    bound_lo = np.array([b[0] for b in bounds])
    bound_hi = np.array([b[1] for b in bounds])

    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    pbar = tqdm(total=n_iters, desc="Adam", unit="step", disable=not VERBOSE)

    start_t = time.time()
    lr_t = lr
    for t in range(1, n_iters + 1):
        # Gradient de -J (evaluate retourne -J)
        grad = _numerical_gradient(
            lambda xx: evaluate(xx, mesh_size=mesh_size), x, h=h_grad,
        )
        # Adam update (minimisation de -J = ascension de J)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x = x - lr_t * m_hat / (np.sqrt(v_hat) + eps)
        x = np.clip(x, bound_lo, bound_hi)
        lr_t *= lr_decay

        pbar.update(1)
        if progress_cb is not None:
            progress_cb(t, n_iters)
    pbar.close()

    elapsed = time.time() - start_t
    # best_J / best_x ont été suivis par evaluate() dans l'historique
    final_best_J = best_J
    final_best_x = best_x.copy() if best_x is not None else x.copy()
    return {
        "method": "Adam",
        "best_J": final_best_J,
        "best_x": final_best_x,
        "x_final": x.copy(),
        "x0": list(x0),
        "n_eval": len(history),
        "time": elapsed,
        "success": True,
        "message": f"{n_iters} Adam steps (lr={lr}, decay={lr_decay})",
        "history": list(history),
    }


def run_adam_then_lbfgs(
    bounds,
    x0=None,
    n_adam_iters: int = 30,
    lr: float = 0.05,
    lr_decay: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    h_grad: float = 1e-3,
    maxiter_lbfgs: int = 50,
    mesh_size: int = 50,
    progress_cb: Optional[Callable[[int, int], None]] = None,
):
    """Hybride deux phases : Adam d'abord (robustesse), L-BFGS-B ensuite (raffinement).

    Phase 1 — Adam pour `n_adam_iters` étapes : amène x dans une bonne
    région grâce à l'adaptation par coordonnée et au momentum.

    Phase 2 — L-BFGS-B à partir de la fin d'Adam : raffine avec une
    convergence superlinéaire et un respect natif des bornes.
    """
    reset_optimization()
    if x0 is None:
        x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])
    else:
        x0 = np.asarray(x0, dtype=float)

    bound_lo = np.array([b[0] for b in bounds])
    bound_hi = np.array([b[1] for b in bounds])

    # ---------- Phase 1 : Adam ----------
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    pbar = tqdm(total=n_adam_iters + 1, desc="Adam→L-BFGS-B",
                unit="step", disable=not VERBOSE)

    start_t = time.time()
    lr_t = lr
    for t in range(1, n_adam_iters + 1):
        grad = _numerical_gradient(
            lambda xx: evaluate(xx, mesh_size=mesh_size), x, h=h_grad,
        )
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x = x - lr_t * m_hat / (np.sqrt(v_hat) + eps)
        x = np.clip(x, bound_lo, bound_hi)
        lr_t *= lr_decay
        pbar.update(1)
        if progress_cb is not None:
            progress_cb(t, n_adam_iters + 1)

    n_eval_after_adam = len(history)
    J_after_adam = float(best_J) if best_J > -np.inf else float("nan")

    # ---------- Phase 2 : L-BFGS-B depuis x ----------
    result_lbfgs = minimize(
        lambda xx: evaluate(xx, mesh_size=mesh_size),
        x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter_lbfgs, "disp": False},
    )
    pbar.update(1)
    pbar.close()

    elapsed = time.time() - start_t
    # Le best global est suivi par evaluate() ; le best peut venir de l'une ou l'autre phase
    final_best_J = best_J
    final_best_x = best_x.copy() if best_x is not None else result_lbfgs.x.copy()

    return {
        "method": "Adam → L-BFGS-B",
        "best_J": final_best_J,
        "best_x": final_best_x,
        "x0": list(x0),
        "x_after_adam": x.tolist(),
        "x_after_lbfgs": result_lbfgs.x.tolist(),
        "J_after_adam": J_after_adam,
        "J_after_lbfgs": -float(result_lbfgs.fun),
        "n_eval": len(history),
        "n_eval_adam": n_eval_after_adam,
        "n_eval_lbfgs": len(history) - n_eval_after_adam,
        "time": elapsed,
        "success": True,
        "message": f"{n_adam_iters} Adam + L-BFGS-B (lr={lr}, decay={lr_decay})",
        "history": list(history),
    }
