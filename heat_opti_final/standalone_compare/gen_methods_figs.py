import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

res = json.load(open("results_methods.json"))
METHODS = ["Nelder-Mead", "Differential Evolution", "Adam → L-BFGS"]
COL = {"Nelder-Mead": "#27ae60", "Differential Evolution": "#e67e22",
       "Adam → L-BFGS": "#2980b9"}
MK = {"Nelder-Mead": "o", "Differential Evolution": "s", "Adam → L-BFGS": "^"}
LAMBDAS = sorted({float(k.split("|")[1]) for k in res})

# ---------- Figure 1 : fronts de Pareto par méthode ----------
fig, ax = plt.subplots(figsize=(8.2, 5.4))
for m in METHODS:
    pts = sorted([(res[f"{m}|{l}"]["cost"], res[f"{m}|{l}"]["J"]) for l in LAMBDAS])
    c = [p[0] for p in pts]; J = [p[1] for p in pts]
    ax.plot(c, J, "-", color=COL[m], lw=1.5, alpha=0.7)
    ax.scatter(c, J, color=COL[m], marker=MK[m], s=80, edgecolors="black",
               linewidths=0.5, label=m, zorder=3)
ax.set_xlabel(r"Coût matériau  $\bar k = \frac{1}{5}\sum_i k_i$")
ax.set_ylabel("Performance  $J$")
ax.set_title("Front de Pareto perf./coût — les trois optimiseurs (mesh 25)")
ax.grid(alpha=0.3); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig("ext_methods_pareto.png", dpi=150); plt.close()

# ---------- Figure 2 : convergence sur F à un lambda représentatif ----------
LAM = 0.05
fig, ax = plt.subplots(figsize=(8.2, 5.4))
for m in METHODS:
    r = res[f"{m}|{LAM}"]
    ax.plot(range(1, len(r["curve"]) + 1), r["curve"], color=COL[m], lw=2,
            label=f"{m} (F*={r['F']:.4f}, {r['n_eval']} év.)")
ax.set_xlabel("Nombre d'évaluations de J")
ax.set_ylabel(r"Meilleur $F_\lambda = J - \lambda\,\bar k$ atteint")
ax.set_title(f"Convergence sur le problème pénalisé  (λ = {LAM}, mesh 25)")
ax.grid(alpha=0.3); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig("ext_methods_convergence.png", dpi=150); plt.close()

# ---------- Tableau récap au lambda représentatif ----------
with open("ext_methods_lambda005.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Méthode", "J", "coût k̄", "F = J-λk̄", "n_eval", "temps (s)"])
    for m in METHODS:
        r = res[f"{m}|{LAM}"]
        w.writerow([m, f"{r['J']:.4f}", f"{r['cost']:.3f}", f"{r['F']:.4f}",
                    r["n_eval"], f"{r['time']:.2f}"])

print("Figures : ext_methods_pareto.png, ext_methods_convergence.png")
print(f"\nRécap à λ={LAM} :")
for m in METHODS:
    r = res[f"{m}|{LAM}"]
    print(f"  {m:24s} J={r['J']:.4f} coût={r['cost']:.3f} F={r['F']:.4f} n_eval={r['n_eval']}")
