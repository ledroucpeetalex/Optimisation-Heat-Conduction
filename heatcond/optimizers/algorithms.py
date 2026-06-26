"""Boîte à outils générique des optimiseurs (couche partagée).

Enveloppes minces autour de bibliothèques éprouvées : **scipy.optimize** pour
Nelder-Mead, Differential Evolution et L-BFGS-B, et **scikit-optimize** pour
l'optimisation bayésienne ; seul Adam est codé à la main (absent des deux).

Ces fonctions s'appliquent à un objectif boîte-noire ``feval(x) -> float`` que l'on
**maximise**, avec des bornes ``bounds`` (liste de ``(lo, hi)``). Elles s'appuient
sur les implémentations éprouvées de SciPy :

* Nelder-Mead            -> ``scipy.optimize.minimize(method="Nelder-Mead")`` ;
* Differential Evolution -> ``scipy.optimize.differential_evolution`` (stratégie,
  popsize, mutation, etc. passés tels quels) ;
* L-BFGS-B               -> ``scipy.optimize.minimize(method="L-BFGS-B")``.

Seul **Adam** est implémenté à la main (gradient par différences finies) car il
n'existe pas dans SciPy ; il sert uniquement de phase d'amorçage de l'hybride
Adam->L-BFGS, dont la phase de raffinement est le L-BFGS-B de SciPy.

Toutes les fonctions agissent sur un objectif boîte-noire ``feval(x) -> float`` à
maximiser, avec des bornes ``bounds``. Cette couche est partagée par les deux modèles (Partie I via
:mod:`heatcond.optimizers.simple`, Partie II via
:mod:`heatcond.optimizers.parametric`). Les variables discrètes (matériaux) sont
gérées par **relaxation + arrondi** : les optimiseurs travaillent en continu,
l'arrondi a lieu dans l'objectif.

``progress_cb(step, total)`` est optionnel et sert aux barres de progression des
interfaces graphiques.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution as _scipy_de


def _lohi(bounds):
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return lo, hi


def nelder_mead(feval, x0, bounds, maxiter=400, progress_cb=None):
    """Simplexe de Nelder-Mead (SciPy), bornes natives."""
    state = {"it": 0}

    def _cb(xk):
        state["it"] += 1
        if progress_cb is not None:
            progress_cb(state["it"], maxiter)

    res = minimize(lambda y: -feval(y), np.asarray(x0, float), method="Nelder-Mead",
                   bounds=bounds, callback=_cb if progress_cb else None,
                   options={"maxiter": maxiter, "disp": False})
    return res.x


def differential_evolution(feval, bounds, popsize=8, maxiter=15,
                           mutation=(0.5, 1.0), recombination=0.7,
                           strategy="best1bin", seed=42, integrality=None,
                           progress_cb=None):
    """Differential Evolution de SciPy (réglages pass-through)."""
    state = {"gen": 0}

    def _cb(xk, convergence=None):
        state["gen"] += 1
        if progress_cb is not None:
            progress_cb(state["gen"], maxiter)
        return False

    res = _scipy_de(lambda y: -feval(y), bounds, popsize=popsize, maxiter=maxiter,
                    mutation=mutation, recombination=recombination, strategy=strategy,
                    seed=seed, polish=False, integrality=integrality, callback=_cb)
    return res.x


def lbfgs_b(feval, x0, bounds, maxiter=50, progress_cb=None):
    """L-BFGS-B de SciPy (gradient estimé par différences finies par SciPy)."""
    state = {"it": 0}

    def _cb(xk):
        state["it"] += 1
        if progress_cb is not None:
            progress_cb(state["it"], maxiter)

    res = minimize(lambda y: -feval(y), np.asarray(x0, float), method="L-BFGS-B",
                   bounds=bounds, callback=_cb if progress_cb else None,
                   options={"maxiter": maxiter, "disp": False})
    return res.x


def _num_grad(f, x, lo, hi, h=1e-3):
    """Gradient central par différences finies (coût 2n évaluations)."""
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        xp = x.copy(); xp[i] = min(x[i] + h, hi[i])
        xm = x.copy(); xm[i] = max(x[i] - h, lo[i])
        d = xp[i] - xm[i]
        if d > 0:
            g[i] = (f(xp) - f(xm)) / d
    return g


def adam(feval, x0, bounds, n_iters=30, lr=0.05, b1=0.9, b2=0.999, eps=1e-8,
         h=1e-3, progress_cb=None):
    """Adam (Kingma & Ba, 2014) avec gradient FD : ascension de ``feval``.

    Absent de SciPy ; sert d'amorçage robuste avant le raffinement L-BFGS-B.
    """
    lo, hi = _lohi(bounds)
    x = np.clip(np.asarray(x0, float), lo, hi)
    m = np.zeros(len(x)); v = np.zeros(len(x))
    for t in range(1, n_iters + 1):
        g = _num_grad(feval, x, lo, hi, h)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        x = np.clip(x + lr * mh / (np.sqrt(vh) + eps), lo, hi)
        if progress_cb is not None:
            progress_cb(t, n_iters)
    return x


def adam_then_lbfgs(feval, x0, bounds, n_adam=30, maxiter_lbfgs=50, lr=0.05,
                    h=1e-3, progress_cb=None):
    """Hybride : amorçage Adam (FD) puis raffinement L-BFGS-B (SciPy)."""
    x_after_adam = adam(feval, x0, bounds, n_iters=n_adam, lr=lr, h=h,
                        progress_cb=progress_cb)
    return lbfgs_b(feval, x_after_adam, bounds, maxiter=maxiter_lbfgs)


# ---------------------------------------------------------------------------
# Optimisation bayésienne (modèle de substitution gaussien) via scikit-optimize
# ---------------------------------------------------------------------------
def bayesian(feval, bounds, n_calls=60, n_initial_points=12, seed=42,
             integrality=None, progress_cb=None):
    """Maximise ``feval`` par optimisation bayésienne (GP + Expected Improvement).

    Construit un modèle de substitution gaussien (Kriging) des points déjà évalués
    et choisit le point suivant par *Expected Improvement* : conçu pour les
    objectifs boîte-noire coûteux évalués en peu d'appels. Les variables discrètes
    (matériaux) sont déclarées comme dimensions ``Integer`` (gestion native du
    mixte, sans relaxation).

    ``scikit-optimize`` est une dépendance **optionnelle** : l'import est paresseux,
    donc le reste du package fonctionne sans elle (``pip install scikit-optimize``).
    """
    from skopt import gp_minimize          # import paresseux (dépendance optionnelle)
    from skopt.space import Real, Integer

    integ = [False] * len(bounds) if integrality is None else list(integrality)
    dims = [Integer(int(lo), int(hi)) if isi else Real(float(lo), float(hi))
            for (lo, hi), isi in zip(bounds, integ)]
    state = {"i": 0}

    def neg(x):
        val = -feval(list(x))
        state["i"] += 1
        if progress_cb is not None:
            progress_cb(state["i"], n_calls)
        return val

    res = gp_minimize(neg, dims, n_calls=n_calls, n_initial_points=n_initial_points,
                      acq_func="EI", random_state=seed)
    return np.asarray(res.x, dtype=float)
