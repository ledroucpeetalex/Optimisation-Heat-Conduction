"""Comparaison Adam->L-BFGS  vs  Differential Evolution  vs  Nelder-Mead
sur le problème HEAT-COND, en numpy pur (le solveur FreeFEM est répliqué
dans fem_solver.py, validé contre les résultats FreeFEM du projet).

Toutes les méthodes MAXIMISENT J. On enregistre l'historique global des
évaluations (J et meilleur-J-courant) pour les courbes de convergence.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # racine du dépôt pour `import heatcond`

import time
import json
import numpy as np
import pandas as pd
from heatcond import config
from heatcond.reference_solver import FEMProblem

BOUNDS = [(0.1, 1.0)] * 5 + [(0.01, 1.0)]
LO = np.array([b[0] for b in BOUNDS])
HI = np.array([b[1] for b in BOUNDS])
X0 = np.array([0.5] * 6)          # design initial canonique du projet
MESH = str(config.CACHE_DIR / "mesh_25.msh")

# ----------------------------------------------------------------------
# Compteur global d'évaluations + historique (meilleur J courant)
# ----------------------------------------------------------------------
class Objective:
    def __init__(self, prob):
        self.prob = prob
        self.reset()

    def reset(self):
        self.n = 0
        self.best = -np.inf
        self.best_x = None
        self.curve = []          # meilleur J courant après chaque éval

    def J(self, x):
        x = np.clip(np.asarray(x, float), LO, HI)
        val = self.prob.solve(x)
        self.n += 1
        if val > self.best:
            self.best = val
            self.best_x = x.copy()
        self.curve.append(self.best)
        return val

    def negJ(self, x):
        return -self.J(x)


def num_grad(f, x, h=1e-3):
    """Gradient central par différences finies (coût 2n évals)."""
    n = len(x); g = np.zeros(n)
    for i in range(n):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


# ----------------------------------------------------------------------
# 1) Adam  (ascension de J, gradient FD, projection sur les bornes)
# ----------------------------------------------------------------------
def adam(obj, x0, n_iters=30, lr=0.05, b1=0.9, b2=0.999, eps=1e-8, h=1e-3):
    x = np.clip(x0.copy(), LO, HI)
    m = np.zeros(6); v = np.zeros(6)
    for t in range(1, n_iters + 1):
        g = num_grad(obj.J, x, h)            # gradient de J (on monte)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        x = x + lr * mh / (np.sqrt(vh) + eps)   # +: maximisation
        x = np.clip(x, LO, HI)
    return x


# ----------------------------------------------------------------------
# 2) L-BFGS projeté sur boîte (two-loop recursion + line search projetée)
#    minimise f = -J
# ----------------------------------------------------------------------
def lbfgs_box(obj, x0, maxiter=50, mem=10, h=1e-3, gtol=1e-6):
    f = obj.negJ
    x = np.clip(x0.copy(), LO, HI)
    g = num_grad(f, x, h)
    S, Y = [], []
    fx = f(x)
    for it in range(maxiter):
        # gradient projeté (critère d'arrêt sur la boîte)
        pg = x - np.clip(x - g, LO, HI)
        if np.linalg.norm(pg) < gtol:
            break
        # ---- direction L-BFGS (two-loop recursion) ----
        q = g.copy(); alpha = []
        for s, y in zip(reversed(S), reversed(Y)):
            rho = 1.0 / (y @ s)
            a = rho * (s @ q); alpha.append(a)
            q = q - a * y
        if S:
            s, y = S[-1], Y[-1]
            gamma = (s @ y) / (y @ y)
        else:
            gamma = 1.0
        r = gamma * q
        for (s, y), a in zip(zip(S, Y), reversed(alpha)):
            rho = 1.0 / (y @ s)
            beta = rho * (y @ r)
            r = r + (a - beta) * s
        d = -r
        if g @ d > 0:                 # pas une direction de descente -> steepest
            d = -g
        # ---- recherche linéaire projetée : meilleur pas d'une séquence géométrique ----
        best_step, best_f, best_x = None, fx, None
        for step in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001):
            xnew = np.clip(x + step * d, LO, HI)
            fnew = f(xnew)
            if fnew < best_f - 1e-12:
                best_step, best_f, best_x = step, fnew, xnew
        if best_step is None:         # aucun pas n'améliore -> convergé
            break
        xnew, fnew = best_x, best_f
        gnew = num_grad(f, xnew, h)
        s = xnew - x; y = gnew - g
        if y @ s > 1e-12:
            S.append(s); Y.append(y)
            if len(S) > mem:
                S.pop(0); Y.pop(0)
        x, g, fx = xnew, gnew, fnew
    return x


def adam_then_lbfgs(obj, x0, n_adam=30, maxiter_lbfgs=50, lr=0.05, h=1e-3):
    x_after_adam = adam(obj, x0, n_iters=n_adam, lr=lr, h=h)
    n_adam_eval = obj.n
    x_final = lbfgs_box(obj, x_after_adam, maxiter=maxiter_lbfgs, h=h)
    return x_final, n_adam_eval


# ----------------------------------------------------------------------
# 3) Nelder-Mead (simplexe, bornes par clipping) — minimise -J
# ----------------------------------------------------------------------
def nelder_mead(obj, x0, maxiter=400, tol=1e-8):
    f = obj.negJ
    n = len(x0)
    # simplexe initial
    sim = [np.clip(x0.copy(), LO, HI)]
    for i in range(n):
        xi = x0.copy()
        step = 0.05 * (HI[i] - LO[i])
        xi[i] = np.clip(xi[i] + step, LO[i], HI[i])
        sim.append(xi)
    sim = np.array(sim)
    fv = np.array([f(s) for s in sim])
    a, ga, be, sg = 1.0, 2.0, 0.5, 0.5
    for _ in range(maxiter):
        idx = np.argsort(fv); sim = sim[idx]; fv = fv[idx]
        if abs(fv[0] - fv[-1]) < tol:
            break
        cen = sim[:-1].mean(axis=0)
        xr = np.clip(cen + a * (cen - sim[-1]), LO, HI); fr = f(xr)
        if fr < fv[0]:
            xe = np.clip(cen + ga * (cen - sim[-1]), LO, HI); fe = f(xe)
            sim[-1], fv[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fv[-2]:
            sim[-1], fv[-1] = xr, fr
        else:
            xc = np.clip(cen + be * (sim[-1] - cen), LO, HI); fc = f(xc)
            if fc < fv[-1]:
                sim[-1], fv[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    sim[i] = np.clip(sim[0] + sg * (sim[i] - sim[0]), LO, HI)
                    fv[i] = f(sim[i])
    return sim[np.argmin(fv)]


# ----------------------------------------------------------------------
# 4) Differential Evolution (rand/1/bin, bornes par clipping)
# ----------------------------------------------------------------------
def differential_evolution(obj, popsize=8, maxiter=15, F=0.7, CR=0.9, seed=42):
    rng = np.random.default_rng(seed)
    n = 6; NP = popsize * n
    pop = rng.uniform(LO, HI, size=(NP, n))
    fit = np.array([obj.J(ind) for ind in pop])    # on maximise J
    for _ in range(maxiter):
        for i in range(NP):
            idxs = [j for j in range(NP) if j != i]
            r1, r2, r3 = rng.choice(idxs, 3, replace=False)
            mut = np.clip(pop[r1] + F * (pop[r2] - pop[r3]), LO, HI)
            cross = rng.random(n) < CR
            if not cross.any():
                cross[rng.integers(n)] = True
            trial = np.where(cross, mut, pop[i])
            ft = obj.J(trial)
            if ft > fit[i]:
                pop[i], fit[i] = trial, ft
    k = np.argmax(fit)
    return pop[k]


# ======================================================================
def run():
    if not Path(MESH).exists():
        print(f"Maillage introuvable : {MESH}\n"
              "Générez-le une fois avec FreeFEM via :  "
              "python -c \"from heatcond.freefem import ensure_mesh; ensure_mesh(25)\"\n"
              "(la comparaison des optimiseurs elle-même ne requiert pas FreeFEM).")
        return None, None
    prob = FEMProblem(MESH)
    print(f"Solveur: mesh={MESH} nv={prob.nv}  "
          f"(validation J0={prob.solve(X0):.6f}, "
          f"J_coin={prob.solve([1,1,1,1,1,0.01]):.6f})\n")

    methods = {}

    # --- Adam -> L-BFGS ---
    obj = Objective(prob); t = time.time()
    xf, n_adam = adam_then_lbfgs(obj, X0, n_adam=30, maxiter_lbfgs=50, lr=0.05)
    methods["Adam -> L-BFGS"] = dict(
        J=obj.best, x=obj.best_x, n_eval=obj.n, time=time.time() - t,
        curve=list(obj.curve), extra=f"{n_adam} évals Adam + {obj.n - n_adam} L-BFGS")

    # --- Nelder-Mead ---
    obj = Objective(prob); t = time.time()
    nelder_mead(obj, X0, maxiter=400)
    methods["Nelder-Mead"] = dict(
        J=obj.best, x=obj.best_x, n_eval=obj.n, time=time.time() - t,
        curve=list(obj.curve), extra="simplexe 6D")

    # --- Differential Evolution ---
    obj = Objective(prob); t = time.time()
    differential_evolution(obj, popsize=8, maxiter=15, seed=42)
    methods["Differential Evolution"] = dict(
        J=obj.best, x=obj.best_x, n_eval=obj.n, time=time.time() - t,
        curve=list(obj.curve), extra="popsize=8, maxiter=15")

    # --- (contexte) Adam seul et L-BFGS seul ---
    obj = Objective(prob); t = time.time()
    adam(obj, X0, n_iters=30, lr=0.05)
    methods["Adam seul"] = dict(J=obj.best, x=obj.best_x, n_eval=obj.n,
                                time=time.time() - t, curve=list(obj.curve),
                                extra="30 itérations")
    obj = Objective(prob); t = time.time()
    lbfgs_box(obj, X0, maxiter=80)
    methods["L-BFGS seul"] = dict(J=obj.best, x=obj.best_x, n_eval=obj.n,
                                  time=time.time() - t, curve=list(obj.curve),
                                  extra="depuis x0=0.5")

    # ---- tableau ----
    J0 = prob.solve(X0)
    rows = []
    for name, r in methods.items():
        rows.append({
            "Méthode": name,
            "J*": round(r["J"], 6),
            "Gain vs J0 (%)": round(100 * (r["J"] - J0) / J0, 1),
            "n_eval": r["n_eval"],
            "Temps (s)": round(r["time"], 3),
            "Temps/éval (ms)": round(1000 * r["time"] / r["n_eval"], 2),
            "Design x* (k1..k5, Bi)": ", ".join(f"{v:.3f}" for v in r["x"]),
            "Note": r["extra"],
        })
    df = pd.DataFrame(rows)
    config.ensure_dirs()
    df.to_csv(config.RESULTS_PART1 / "part1_methods_freefemless.csv", index=False)
    with open(config.RESULTS_PART1 / "part1_curves_freefemless.json", "w") as fjs:
        json.dump({k: v["curve"] for k, v in methods.items()}, fjs)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(df.to_string(index=False))
    print(f"\nJ0 (design uniforme 0.5) = {J0:.6f}")
    return methods, df


def plot_convergence(methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in methods.items():
        ax.plot(range(1, len(r["curve"]) + 1), r["curve"], lw=2, label=name)
    ax.set_xlabel("Nombre d'évaluations de J")
    ax.set_ylabel("Meilleur J atteint")
    ax.set_title("Comparaison des optimiseurs — solveur NumPy (sans FreeFEM), maillage 25")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout()
    out = config.RESULTS_PART1 / "part1_convergence_freefemless.png"
    fig.savefig(out, dpi=150)
    print(f"Figure : {out}")


if __name__ == "__main__":
    methods, df = run()
    if methods:
        plot_convergence(methods)