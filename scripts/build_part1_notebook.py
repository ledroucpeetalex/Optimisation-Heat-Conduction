"""Generate `notebooks/part1_simple_objective.ipynb`: the studies of the
initial HEAT-COND problem (maximize the mean fin temperature J over k1..k5 and Bi),
run with SciPy optimizers on the validated pure-NumPy reference solver (no FreeFEM), English labels.

    python scripts/build_part1_notebook.py
"""
import json
import os

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

md(r"""# Initial HEAT-COND problem (Part I) — optimizer comparison and studies

We maximize the mean temperature on the fin boundary,
$$J(\mathbf{x})=\frac{1}{|\Gamma_{\text{fin}}|}\int_{\Gamma_{\text{fin}}}T\,\mathrm{d}\Gamma,$$
over $\mathbf{x}=(k_1,\dots,k_5,\mathrm{Bi})$ at fixed geometry. The studies below
compare four optimizers (Nelder-Mead, Differential Evolution, Adam$\to$L-BFGS, Bayesian optimization)
and probe mesh sensitivity, robustness, and the gradient hybrid. Everything runs on the validated pure-NumPy reference solver (no FreeFEM needed); at
the optimal corner it reproduces the FreeFEM value $J=0.730$.""")

code(r"""
import os, sys, json, time
_root = os.path.abspath(".")
if not os.path.isdir(os.path.join(_root, "heatcond")):
    _root = os.path.abspath("..")
sys.path.insert(0, _root)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from heatcond import config
from heatcond.objective import SimpleObjective
from heatcond.optimizers import algorithms as so
from heatcond.reference_solver_param import build_comb_mesh, solve_param, solve_param_field

config.ensure_dirs()
R = config.RESULTS_PART1
LO = np.array([0.1] * 5 + [0.01]); HI = np.array([1.0] * 5 + [1.0])
BND = [(0.1, 1.0)] * 5 + [(0.01, 1.0)]
X0 = np.array([0.5] * 6)
PARAM = ["k1", "k2", "k3", "k4", "k5", "Bi"]
COL = {"Nelder-Mead": "#27ae60", "Differential Evolution": "#e67e22", "Adam -> L-BFGS": "#2980b9"}

def sobj(mesh=25):
    return SimpleObjective(mesh_size=mesh)

def run_method(name, mesh=25, seed=42, x0=X0):
    obj = sobj(mesh); t0 = time.time()
    if name == "Nelder-Mead":
        so.nelder_mead(obj.J, x0, BND, maxiter=400)
    elif name == "Differential Evolution":
        so.differential_evolution(obj.J, BND, popsize=6, maxiter=12, seed=seed)
    else:
        so.adam_then_lbfgs(obj.J, x0, BND, n_adam=30, maxiter_lbfgs=50, lr=0.05)
    return dict(method=name, best_J=obj.best, best_x=obj.best_x.tolist(),
                n_eval=obj.n, time=time.time() - t0, curve=list(obj.curve))

print("corner J at mesh 50:", round(solve_param([1]*5, [0.06]*5, [0.45]*5, 0.01, 50)[1], 6))
print("J0 (uniform 0.5) at mesh 25:", round(solve_param([0.5]*5, [0.06]*5, [0.45]*5, 0.5, 25)[1], 6))
""")

md(r"""## Geometry and mesh""")
code(r"""
V, Tr, region, E = build_comb_mesh([0.06]*5, [0.45]*5, 25)
fig, ax = plt.subplots(figsize=(6, 6))
ax.triplot(mtri.Triangulation(V[:,0], V[:,1], Tr[:,:3]), "-", lw=0.4, color="#888")
for lbl, color, name in [(1, "#c0392b", "Base (T = 1, Dirichlet)"), (2, "#2980b9", "Fin boundary (Robin)")]:
    seg = E[E[:,2] == lbl]
    for a, b, _ in seg:
        ax.plot([V[a,0], V[b,0]], [V[a,1], V[b,1]], "-", color=color, lw=1.8)
    ax.plot([], [], "-", color=color, lw=2.5, label=name)
ax.set_aspect("equal"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(f"HEAT-COND geometry and $P_1$ mesh ({len(V)} vertices, {len(Tr)} triangles)")
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
fig.tight_layout(); fig.savefig(R / "geometry_mesh.png", dpi=140); plt.show()
""")

md(r"""## Study 1 — Comparison of the methods and the optimal design (mesh 25)""")
code(r"""
s1 = {m: run_method(m, mesh=25) for m in ["Nelder-Mead", "Differential Evolution", "Adam -> L-BFGS"]}
COL["Bayesian opt."] = "#8e44ad"
try:
    _o = sobj(25); _t = time.time()
    so.bayesian(_o.J, BND, n_calls=60, n_initial_points=12, seed=42)
    s1["Bayesian opt."] = dict(method="Bayesian opt.", best_J=_o.best, best_x=_o.best_x.tolist(),
                               n_eval=_o.n, time=time.time() - _t, curve=list(_o.curve))
except ImportError:
    print("scikit-optimize not installed; Bayesian optimization omitted (pip install scikit-optimize).")
rows = [dict(Method=m, Jstar=r["best_J"], n_eval=r["n_eval"], time_s=round(r["time"],3),
            design=", ".join(f"{v:.3f}" for v in r["best_x"])) for m, r in s1.items()]
df1 = pd.DataFrame(rows).sort_values("Jstar", ascending=False)
print(df1.to_string(index=False))

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for m, r in s1.items():
    ax.plot(range(1, len(r["curve"])+1), r["curve"], lw=2, color=COL[m], label=m)
ax.set_xlabel("Function evaluations"); ax.set_ylabel("Best $J$ so far")
ax.set_title("Convergence of the methods (mesh 25)"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(R / "part1_methods_convergence.png", dpi=140)
# standalone quality-vs-cost figure
fig2, axp = plt.subplots(figsize=(5.8, 4.4))
for m, r in s1.items():
    axp.scatter(r["n_eval"], r["best_J"], s=110, color=COL[m], edgecolors="black", linewidths=0.5, label=m, zorder=3)
axp.set_xlabel("Cost (number of evaluations)"); axp.set_ylabel("$J^\\star$ reached")
axp.set_title("Quality vs cost"); axp.legend(loc="lower left"); axp.grid(alpha=0.3)
fig2.tight_layout(); fig2.savefig(R / "part1_quality_vs_cost.png", dpi=140); plt.show()
""")

md(r"""## Temperature fields — initial vs optimized designs (mesh 50)""")
code(r"""
designs = {"Initial (x = 0.5)": X0}
for m, r in s1.items():
    designs[m] = np.array(r["best_x"])
fields = {}
vmin, vmax = 1.0, 0.0
for name, x in designs.items():
    V, Tr, T = solve_param_field(x[:5], [0.06]*5, [0.45]*5, float(x[5]), 50)
    fields[name] = (V, Tr, T); vmin = min(vmin, T.min()); vmax = max(vmax, T.max())
fig, axes = plt.subplots(1, len(designs), figsize=(4 * len(designs), 4.4))
for ax, (name, (V, Tr, T)) in zip(axes, fields.items()):
    tcf = ax.tricontourf(mtri.Triangulation(V[:,0], V[:,1], Tr[:,:3]), T,
                         levels=np.linspace(vmin, vmax, 25), cmap="inferno")
    ax.set_aspect("equal"); ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
    ax.set_title(name, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(tcf, ax=axes, fraction=0.025, pad=0.02, label="T")
fig.suptitle("Temperature field: initial vs optimized designs (shared colour scale, mesh 50)")
fig.savefig(R / "part1_temperature_fields.png", dpi=140); plt.show()
print("min fin T initial -> optimized:", round(fields["Initial (x = 0.5)"][2].min(),3),
      "->", round(min(fields[m][2].min() for m in s1),3))
""")

md(r"""## Study 2 — Mesh sensitivity (Nelder-Mead)""")
code(r"""
meshes = [15, 25, 40, 60]
Js, ts = [], []
for ms in meshes:
    r = run_method("Nelder-Mead", mesh=ms)
    Js.append(r["best_J"]); ts.append(r["time"])
print(pd.DataFrame({"mesh": meshes, "Jstar": np.round(Js,4), "time_s": np.round(ts,2)}).to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(meshes, Js, "o-", lw=2); ax[0].set_xlabel("Mesh density"); ax[0].set_ylabel("$J^\\star$")
ax[0].set_title("Mesh sensitivity of $J^\\star$"); ax[0].grid(alpha=0.3)
ax[1].plot(meshes, ts, "s-", lw=2, color="C3"); ax[1].set_xlabel("Mesh density"); ax[1].set_ylabel("Optimization time (s)")
ax[1].set_title("Computational cost"); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(R / "part1_mesh_sensitivity.png", dpi=140); plt.show()
""")

md(r"""## Studies 3 and 4 — Robustness (starting point for NM/Adam, seed for DE)""")
code(r"""
rng = np.random.default_rng(0)
starts = [rng.uniform(LO, HI) for _ in range(5)]
nm_J = [run_method("Nelder-Mead", mesh=25, x0=x0)["best_J"] for x0 in starts]
ad_J = [run_method("Adam -> L-BFGS", mesh=25, x0=x0)["best_J"] for x0 in starts]
de_J = [run_method("Differential Evolution", mesh=25, seed=sd)["best_J"] for sd in range(5)]
data = {"Nelder-Mead\n(starts)": nm_J, "Adam->L-BFGS\n(starts)": ad_J, "Diff. Evolution\n(seeds)": de_J}
gcol = {"Nelder-Mead\n(starts)": COL["Nelder-Mead"], "Adam->L-BFGS\n(starts)": COL["Adam -> L-BFGS"],
        "Diff. Evolution\n(seeds)": COL["Differential Evolution"], "Bayesian\n(seeds)": "#8e44ad"}
try:
    bo_J = []
    for sd in range(5):
        _ob = sobj(25); so.bayesian(_ob.J, BND, n_calls=50, n_initial_points=10, seed=sd); bo_J.append(_ob.best)
    data["Bayesian\n(seeds)"] = bo_J
except ImportError:
    pass
print({k: (round(np.mean(v),4), round(np.std(v),5)) for k, v in data.items()})
fig, ax = plt.subplots(figsize=(8.0, 4.6))
for i, (k, v) in enumerate(data.items()):
    ax.scatter([i]*len(v), v, s=60, alpha=0.7, color=gcol[k])
    ax.errorbar(i, np.mean(v), yerr=np.std(v), fmt="_", color="black", capsize=8, ms=20)
ax.set_xticks(range(len(data))); ax.set_xticklabels(list(data.keys()))
ax.set_ylabel("$J^\\star$"); ax.set_title("Robustness: spread of $J^\\star$ across starts / seeds"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(R / "part1_robustness.png", dpi=140); plt.show()
""")

md(r"""## Study 7 — Adam$\to$L-BFGS hybrid and preconditioning, and optimal designs""")
code(r"""
# diagnostic runs from x0 = 0.5
o_h = sobj(25); so.adam_then_lbfgs(o_h.J, X0, BND, n_adam=30, maxiter_lbfgs=50, lr=0.05); hyb=(o_h.best, o_h.n, list(o_h.curve))
o_a = sobj(25); so.adam(o_a.J, X0, BND, n_iters=30, lr=0.05); adamonly=(o_a.best, o_a.n)
o_l = sobj(25); so.lbfgs_b(o_l.J, X0, BND, maxiter=80); lbfgsonly=(o_l.best, o_l.n, o_l.best_x.tolist(), list(o_l.curve))
print("Adam->L-BFGS J=%.6f (n=%d) | Adam alone J=%.6f (n=%d) | L-BFGS alone J=%.6f (n=%d)"
      % (hyb[0], hyb[1], adamonly[0], adamonly[1], lbfgsonly[0], lbfgsonly[1]))
print("L-BFGS alone design:", np.round(lbfgsonly[2],3))

w = 0.25; xpos = np.arange(6)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(range(1,len(hyb[2])+1), hyb[2], lw=2, color=COL["Adam -> L-BFGS"], label="Adam -> L-BFGS")
ax.plot(range(1,len(s1["Nelder-Mead"]["curve"])+1), s1["Nelder-Mead"]["curve"], lw=2, color=COL["Nelder-Mead"], label="Nelder-Mead")
ax.plot(range(1,len(s1["Differential Evolution"]["curve"])+1), s1["Differential Evolution"]["curve"], lw=2, color=COL["Differential Evolution"], label="Differential Evolution")
ax.plot(range(1,len(lbfgsonly[3])+1), lbfgsonly[3], lw=2, ls="--", color="gray", label="L-BFGS alone")
ax.set_xlabel("Function evaluations"); ax.set_ylabel("Best $J$ so far")
ax.set_title("Gradient methods: the hybrid, its components, vs NM and DE"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(R / "part1_gradient_methods.png", dpi=140)
# standalone designs bar
fig3, axd = plt.subplots(figsize=(6.0, 4.4))
for i, m in enumerate(["Nelder-Mead","Differential Evolution","Adam -> L-BFGS"]):
    axd.bar(xpos + (i-1)*w, s1[m]["best_x"], w, color=COL[m], label=m)
axd.set_xticks(xpos); axd.set_xticklabels(PARAM); axd.set_ylabel("Value")
axd.set_title("Optimal designs found by the methods"); axd.legend(fontsize=8); axd.grid(alpha=0.3, axis="y")
fig3.tight_layout(); fig3.savefig(R / "part1_optimal_designs.png", dpi=140); plt.show()
""")

code(r"""
# Save the key numbers so the report tables can be updated from this run.
summary = {
    "corner_J_mesh50": float(solve_param([1]*5, [0.06]*5, [0.45]*5, 0.01, 50)[1]),
    "J0_uniform_mesh25": float(solve_param([0.5]*5, [0.06]*5, [0.45]*5, 0.5, 25)[1]),
    "study1": {m: {"best_J": r["best_J"], "n_eval": r["n_eval"], "best_x": r["best_x"]} for m, r in s1.items()},
    "mesh_sensitivity": {"density": meshes, "Jstar": [float(v) for v in Js]},
    "hybrid": {
        "adam_then_lbfgs_J": float(hyb[0]),
        "adam_alone_J": float(adamonly[0]),
        "lbfgs_alone_J": float(lbfgsonly[0]), "lbfgs_alone_n": int(lbfgsonly[1]),
    },
}
json.dump(summary, open(R / "summary.json", "w"), indent=2)
print("Saved", R / "summary.json")
""")

md(r"""### Summary

All three methods reach the same boundary optimum $\mathbf{x}^\star=(1,1,1,1,1,0.01)$ with
$J^\star\approx0.73$ (mesh 25). Nelder-Mead is the cheapest; Differential Evolution serves as a
global-optimality check; the Adam$\to$L-BFGS hybrid shows that adaptive preconditioning rescues a
gradient method where plain L-BFGS stalls. Figures saved to `results/part1/`.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3"}},
      "nbformat": 4, "nbformat_minor": 5}
_here = os.path.dirname(os.path.abspath(__file__))
_out = os.path.join(_here, "..", "notebooks", "part1_simple_objective.ipynb")
with open(_out, "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("part1_simple_objective.ipynb generated:", len(cells), "cells")
