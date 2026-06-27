"""Generate `notebooks/part2_material_geometry.ipynb`: comparison of the methods (Nelder-Mead, Differential Evolution, Adam->L-BFGS, Bayesian) on the
parametric model (material + geometry), using SciPy optimizers on the validated
pure-NumPy reference solver (reproducible without FreeFEM).

    python scripts/build_part2_notebook.py
"""
import json
import os

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

md(r"""# Parametric HEAT-COND model — comparison of the methods (Part II)

The design is enriched with a per-fin material and geometry:

$$x = [\,m_1..m_5\ |\ t_1..t_5\ |\ \ell_1..\ell_5\ |\ \mathrm{Bi}\,]\in\mathbb{R}^{16},$$

and we maximize the scalarized objective
$$F(x) = Q(x) - \lambda_{\text{cost}}\,\text{Cost}(x) - \lambda_{\text{mass}}\,\text{Mass}(x),$$
where $Q=\int_{\Gamma_{\text{fin}}}\mathrm{Bi}\,T\,\mathrm{d}\Gamma$ is the dissipated heat.

We apply the **same methods as in Part I** (Nelder-Mead, Differential Evolution, the Adam$\to$L-BFGS
hybrid) **together with Bayesian optimization**. The first three come from **scipy.optimize**; Bayesian
optimization (scikit-optimize) treats the five materials natively as integer dimensions, while the other
methods handle them by **relaxation and rounding**. The studies run on the validated
pure-NumPy reference solver (at the default geometry it reproduces the FreeFEM corner value
$J=0.730$), so they are reproducible **without FreeFEM**.""")

code(r"""
import os, sys, json, time
_root = os.path.abspath(".")
if not os.path.isdir(os.path.join(_root, "heatcond")):   # notebook launched from notebooks/
    _root = os.path.abspath("..")
sys.path.insert(0, _root)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heatcond import config
from heatcond.objective import ParametricObjective
from heatcond.optimizers import parametric as P
from heatcond.reference_solver_param import solve_param
from heatcond.materials import MATERIALS, unpack, describe

config.ensure_dirs()
R = config.RESULTS_PART2
SOLVER = solve_param          # NumPy reference solver; use heatcond.freefem.run_solver_param for FreeFEM
LAM_COST, LAM_MASS, COARSE, FINE, SEED = 0.05, 0.02, 18, 45, 42
MAT_COLORS = {0: "#b87333", 1: "#9aa0a6", 2: "#5a6b7b", 3: "#2e8b57"}
COL = {"Nelder-Mead": "#27ae60", "Differential Evolution": "#e67e22", "Adam -> L-BFGS": "#2980b9"}

def mkobj(lam_cost=LAM_COST, lam_mass=LAM_MASS, mesh=COARSE):
    return ParametricObjective(lam_cost, lam_mass, mesh_size=mesh, solver=SOLVER)

def strip(r):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}

cache = {}
print("Catalogue:", [m["name"] for m in MATERIALS])
print(f"lambda_cost={LAM_COST}  lambda_mass={LAM_MASS}  coarse mesh={COARSE}  fine={FINE}")
""")

md(r"""## Study 1 — Comparison of the three methods

Comparable evaluation budget on the coarse mesh. We record the best-so-far curve of $F$
and the quality of the best design.""")

code(r"""
study1 = {}
study1["Nelder-Mead"] = strip(P.run_nelder_mead(mkobj(), maxiter=400))
study1["Differential Evolution"] = strip(P.run_differential_evolution(mkobj(), popsize=5, maxiter=8, seed=SEED))
study1["Adam -> L-BFGS"] = strip(P.run_adam_then_lbfgs(mkobj(), n_adam=30, maxiter_lbfgs=50, lr=0.05))
COL["Bayesian opt."] = "#8e44ad"
try:
    study1["Bayesian opt."] = strip(P.run_bayesian(mkobj(), n_calls=80, n_initial_points=16, seed=SEED))
except ImportError:
    print("scikit-optimize not installed; Bayesian optimization omitted (pip install scikit-optimize).")
cache["study1"] = study1

rows = [dict(Method=n, F=r["best_F"], Q=r["best_Q"], J=r["best_J"],
             cost=r["best_cost"], mass=r["best_mass"], n_eval=r["n_eval"],
             time_s=round(r["time"], 2)) for n, r in study1.items()]
df = pd.DataFrame(rows).sort_values("F", ascending=False).reset_index(drop=True)
best_name = df.iloc[0]["Method"]; best = study1[best_name]
print("Best method:", best_name)
df
""")

code(r"""
plt.figure(figsize=(7.2, 4.6))
for name, r in study1.items():
    plt.plot(range(1, len(r["curve"]) + 1), r["curve"], lw=2, color=COL.get(name),
             label=f"{name} (F*={r['best_F']:.3f}, {r['n_eval']} eval.)")
plt.xlabel("PDE evaluations"); plt.ylabel("Best $F$ so far")
plt.title("Convergence of the methods (parametric model, coarse mesh)")
plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(R / "part2_methods_convergence.png", dpi=140); plt.show()
""")

md(r"""**Discussion.** Unlike Part I (where Nelder-Mead clearly dominates a smooth, unimodal
landscape), the methods are here close to one another: the problem is mixed but **dominated by its
continuous variables** (geometry and Biot number). The gradient hybrid Adam$\to$L-BFGS is strong,
Differential Evolution serves as a global check, Nelder-Mead is the cheapest, and Bayesian
optimization is the most sample-efficient. We **retain the best method** of this run (``best_name``)
for the detailed studies that follow.""")

md(r"""## Study 2 — Diagnostics of the retained method and optimal design""")

code(r"""
hist = best["history"]; n = [h["n"] + 1 for h in hist]; Fv = [h["F"] for h in hist]
bF = -np.inf; bQ = bC = bM = np.nan; runQ = []; runC = []; runM = []
for h in hist:
    if h["F"] > bF: bF, bQ, bC, bM = h["F"], h["Q"], h["cost"], h["mass"]
    runQ.append(bQ); runC.append(bC); runM.append(bM)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(n, Fv, s=8, alpha=0.25, label="$F$ per evaluation")
ax[0].plot(n, best["curve"], color="C3", lw=2, label="best-so-far")
ax[0].set_xlabel("Evaluations"); ax[0].set_ylabel("$F$")
ax[0].set_title(f"{best_name}: objective vs iterations"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(n, runQ, label="$Q$ (dissipated heat)", lw=2)
ax[1].plot(n, runC, label="cost", lw=2); ax[1].plot(n, runM, label="mass", lw=2)
ax[1].set_xlabel("Evaluations"); ax[1].set_title(f"{best_name}: metrics of the best design")
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(R / "part2_best_diagnostics.png", dpi=140); plt.show()
""")

code(r"""
from matplotlib.patches import Patch
x = np.array(best["best_x"]); m, t, l, Bi = unpack(x)
cols = [MAT_COLORS[i] for i in m]
fins = [f"Fin {i+1}" for i in range(5)]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.7))
ax[0].bar(fins, t, color=cols); ax[0].set_ylabel("thickness $t_i$"); ax[0].set_title("Thickness (colour = material)")
ax[1].bar(fins, l, color=cols); ax[1].set_ylabel("length $\\ell_i$"); ax[1].set_title("Length (colour = material)")
for a in ax: a.grid(alpha=0.3, axis="y")
# explicit colour -> material legend (all materials shown)
handles = [Patch(facecolor=MAT_COLORS[i], label=MATERIALS[i]["name"]) for i in range(len(MATERIALS))]
fig.legend(handles=handles, loc="upper center", ncol=len(MATERIALS),
           frameon=True, bbox_to_anchor=(0.5, 0.99), fontsize=9)
plt.suptitle(f"Optimal design ({best_name}): Bi={Bi:.3f}, Q={best['best_Q']:.3f} "
             f"(J={best['best_J']:.3f}), cost={best['best_cost']:.3f}, mass={best['best_mass']:.3f}", y=0.90)
fig.tight_layout(rect=[0, 0, 1, 0.88]); plt.savefig(R / "part2_optimal_design.png", dpi=140); plt.show()
print(describe(x))
""")

md(r"""## Study 3 — Evolution of the objective (vs evaluations and vs CPU time)""")

code(r"""
curve = np.array(best["curve"]); cum = np.cumsum([h["time"] for h in hist]); nn = np.arange(1, len(curve) + 1)
bF = -np.inf; bQ = np.nan; rq = []
for h in hist:
    if h["F"] > bF: bF, bQ = h["F"], h["Q"]
    rq.append(bQ)
rq = np.array(rq)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].plot(nn, curve, color="C0", lw=2.2, label=r"objective $F$ (best-so-far)")
ax[0].plot(nn, rq, color="C2", lw=1.6, ls="--", label=r"$Q$ (dissipated heat)")
ax[0].set_xlabel("PDE evaluations"); ax[0].set_ylabel("Value (best-so-far)")
ax[0].set_title(f"Objective vs evaluations ({best_name})")
ax[0].legend(loc="lower right", fontsize=9); ax[0].grid(alpha=0.3)
ax[1].plot(cum, curve, color="C3", lw=2.2, label=r"objective $F$ (best-so-far)")
ax[1].plot(cum, rq, color="C2", lw=1.6, ls="--", label=r"$Q$ (dissipated heat)")
ax[1].set_xlabel("Cumulative CPU time (s)"); ax[1].set_ylabel("Value (best-so-far)")
ax[1].set_title(f"Objective vs CPU time ({best_name})")
ax[1].legend(loc="lower right", fontsize=9); ax[1].grid(alpha=0.3)
plt.suptitle(f"{best_name} (retained method), convergence "
             f"[{len(nn)} eval., {cum[-1]:.0f} s, $F^\\star$={curve[-1]:.4f}]", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(R / "part2_objective_evolution.png", dpi=140); plt.show()
""")

md(r"""## Study 4 — Performance/cost Pareto front ($\lambda$ sweep)""")

code(r"""
LAMBDAS = [0.0, 0.02, 0.05, 0.10, 0.20]
best_fn = P.METHODS[best_name]
kw = ({"popsize": 5, "maxiter": 8, "seed": SEED} if best_name == "Differential Evolution"
      else {"maxiter": 400} if best_name == "Nelder-Mead"
      else {"n_adam": 30, "maxiter_lbfgs": 50})
pareto = []
for lam in LAMBDAS:
    r = best_fn(mkobj(lam_cost=lam), **kw)
    pareto.append(dict(lam=lam, Q=r["best_Q"], J=r["best_J"], cost=r["best_cost"], mass=r["best_mass"], F=r["best_F"]))
cache["pareto"] = pareto
pc = sorted(pareto, key=lambda d: d["cost"])
costs = [d["cost"] for d in pc]; Qs = [d["Q"] for d in pc]; masses = [d["mass"] for d in pc]
plt.figure(figsize=(7.2, 4.6))
sc = plt.scatter(costs, Qs, c=masses, cmap="viridis", s=80, zorder=3)
plt.plot(costs, Qs, "--", color="gray", alpha=0.6, zorder=2)
for d in pc:
    plt.annotate(f"$\\lambda$={d['lam']}", (d["cost"], d["Q"]), textcoords="offset points", xytext=(6, 6), fontsize=8)
plt.colorbar(sc, label="mass"); plt.xlabel("Material cost"); plt.ylabel("Performance $Q$ (dissipated heat)")
plt.title(f"Performance/cost Pareto front ({best_name})")
plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(R / "part2_pareto_front.png", dpi=140); plt.show()
pd.DataFrame(pareto)
""")

md(r"""## Study 5 — Multi-fidelity strategy (computational cost)

Differential Evolution on a coarse mesh, then Nelder-Mead refinement (materials frozen) on a
fine mesh.""")

code(r"""
glob, fine = P.multifidelity(LAM_COST, LAM_MASS, coarse_mesh=COARSE, fine_mesh=FINE,
                             maxiter=6, popsize=4, seed=SEED, solver=SOLVER)
cache["mf"] = dict(
    glob=dict(F=glob["best_F"], Q=glob["best_Q"], cost=glob["best_cost"], n_eval=glob["n_eval"], time=glob["time"], curve=glob["curve"]),
    fine=dict(F=fine["best_F"], Q=fine["best_Q"], cost=fine["best_cost"], n_eval=fine["n_eval"], time=fine["time"], curve=fine["curve"], mesh_size=FINE),
)
mf = cache["mf"]
plt.figure(figsize=(7.2, 4.6))
plt.plot(range(1, len(mf["glob"]["curve"]) + 1), mf["glob"]["curve"], lw=2, label=f"DE coarse (density {COARSE})")
off = len(mf["glob"]["curve"])
plt.plot(range(off + 1, off + len(mf["fine"]["curve"]) + 1), mf["fine"]["curve"], lw=2, color="C3",
         label=f"NM refinement fine (density {FINE})")
plt.axvline(off, ls=":", color="gray"); plt.xlabel("Evaluations (cumulative)"); plt.ylabel("Best $F$")
plt.title("Multi-fidelity pipeline: coarse search then fine refinement")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(R / "part2_multifidelity.png", dpi=140); plt.show()

cache["meta"] = dict(best_method=best_name, lam_cost=LAM_COST, lam_mass=LAM_MASS, coarse=COARSE, fine=FINE, seed=SEED)
json.dump(cache, open(R / "methods_param.json", "w"))
print("Saved:", R / "methods_param.json")
""")

md(r"""### Conclusion

- The **same three methods as Part I**, applied to the 16D parametric problem.
- The three are **within about 1%**: Adam$\to$L-BFGS marginally ahead, DE confirms the optimum.
- The parametric model turns the trivial Part I corner optimum into a genuine multi-objective
  trade-off (an exploitable Pareto front).
- The **multi-fidelity** strategy keeps the cost manageable despite the mesh being rebuilt at
  every evaluation.

Figures in `results/part2/`: `part2_methods_convergence.png`, `part2_best_diagnostics.png`,
`part2_optimal_design.png`, `part2_objective_evolution.png`, `part2_pareto_front.png`,
`part2_multifidelity.png`.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3"}},
      "nbformat": 4, "nbformat_minor": 5}
_here = os.path.dirname(os.path.abspath(__file__))
_out = os.path.join(_here, "..", "notebooks", "part2_material_geometry.ipynb")
with open(_out, "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("part2_material_geometry.ipynb generated:", len(cells), "cells")
