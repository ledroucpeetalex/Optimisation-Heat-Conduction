"""Étude : les TROIS optimiseurs sur le problème pénalisé (coût matériau).

Pour chaque lambda et chaque méthode (NM, DE, Adam->L-BFGS), on maximise
F_lambda(x) = J(x) - lambda * mean(k_i) et on enregistre (J, coût, F, n_eval,
temps, courbe). Sauvegarde incrémentale dans results_methods.json (reprise
possible si le run est coupé).
"""
import json, os, time
import numpy as np
from fem_solver import FEMProblem
from compare_opt import nelder_mead, differential_evolution, adam_then_lbfgs, LO, HI

MESH = "mesh_25.msh"
X0 = np.array([0.5] * 6)
LAMBDAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.15]
METHODS = ["Nelder-Mead", "Differential Evolution", "Adam → L-BFGS"]
OUT = "results_methods.json"

prob = FEMProblem(MESH)


class PenObjective:
    def __init__(self, prob, lam):
        self.prob = prob; self.lam = lam
        self.n = 0; self.best = -np.inf; self.best_x = None
        self.bestJ = None; self.bestC = None; self.curve = []

    def J(self, x):
        x = np.clip(np.asarray(x, float), LO, HI)
        trueJ = self.prob.solve(x); C = float(np.mean(x[:5]))
        F = trueJ - self.lam * C
        self.n += 1
        if F > self.best:
            self.best, self.best_x, self.bestJ, self.bestC = F, x.copy(), trueJ, C
        self.curve.append(self.best)
        return F

    def negJ(self, x):
        return -self.J(x)


def run_method(method, lam):
    o = PenObjective(prob, lam)
    t = time.time()
    if method == "Nelder-Mead":
        nelder_mead(o, X0, maxiter=250)
    elif method == "Differential Evolution":
        differential_evolution(o, popsize=6, maxiter=8, seed=42)
    else:
        adam_then_lbfgs(o, X0, n_adam=20, maxiter_lbfgs=25)
    return dict(J=o.bestJ, cost=o.bestC, F=o.best, n_eval=o.n,
                time=time.time() - t, x=list(o.best_x), curve=list(o.curve))


results = json.load(open(OUT)) if os.path.exists(OUT) else {}
for method in METHODS:
    for lam in LAMBDAS:
        key = f"{method}|{lam}"
        if key in results:
            continue
        r = run_method(method, lam)
        results[key] = r
        json.dump(results, open(OUT, "w"))
        print(f"{method:24s} λ={lam:5.3f}  J={r['J']:.4f}  coût={r['cost']:.3f}  "
              f"F={r['F']:.4f}  n_eval={r['n_eval']}")

print(f"\nTerminé : {len(results)}/{len(METHODS)*len(LAMBDAS)} configurations.")
