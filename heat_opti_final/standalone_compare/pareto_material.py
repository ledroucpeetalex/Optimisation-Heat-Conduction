"""Extension 'coût matériau' : compromis performance / quantité de matériau.

On maximise l'objectif scalarisé
    F_lambda(x) = J(x) - lambda * cbar,   cbar = (1/5) * sum_i k_i
pour une grille de lambda >= 0. Pour chaque lambda on optimise les 6 variables
(k1..k5, Bi) par Nelder-Mead, et on enregistre le point (cbar, J) atteint :
c'est un point du front de Pareto performance vs coût.

lambda = 0  -> coin (tout au max, J maximal, coût maximal).
lambda grand -> on sacrifie du J pour réduire le coût matériau.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fem_solver import FEMProblem
from compare_opt import nelder_mead, LO, HI

MESH = "mesh_25.msh"
prob = FEMProblem(MESH)
LABELS = ['k1', 'k2', 'k3', 'k4', 'k5', 'Bi']


class PenObj:
    """Objectif pénalisé pour Nelder-Mead (qui MINIMISE negJ = -(J - lambda*cbar))."""
    def __init__(self, prob, lam):
        self.prob = prob; self.lam = lam; self.n = 0
        self.bestF = -np.inf; self.best_x = None; self.bestJ = None; self.bestC = None

    def negJ(self, x):
        x = np.clip(np.asarray(x, float), LO, HI)
        J = self.prob.solve(x)
        cbar = float(np.mean(x[:5]))
        F = J - self.lam * cbar
        self.n += 1
        if F > self.bestF:
            self.bestF, self.best_x, self.bestJ, self.bestC = F, x.copy(), J, cbar
        return -F


def optimize(lam, n_restarts=3, seed=0):
    """NM multi-départ sur l'objectif pénalisé (robustesse)."""
    rng = np.random.default_rng(seed)
    best = None
    starts = [np.array([0.9]*5 + [0.01]), np.array([0.5]*5 + [0.05])]
    starts += [rng.uniform(LO, HI) for _ in range(n_restarts)]
    for x0 in starts:
        o = PenObj(prob, lam)
        nelder_mead(o, x0, maxiter=200)
        if best is None or o.bestF > best.bestF:
            best = o
    return best


# --- balayage de lambda ---
lambdas = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5]
rows = []
for lam in lambdas:
    o = optimize(lam)
    x = o.best_x
    rows.append(dict(lam=lam, cbar=o.bestC, J=o.bestJ,
                     k1=x[0], k2=x[1], k3=x[2], k4=x[3], k5=x[4], Bi=x[5]))
    print(f"lam={lam:5.3f}  cbar={o.bestC:.3f}  J={o.bestJ:.4f}  "
          f"k=[{', '.join(f'{v:.2f}' for v in x[:5])}]  Bi={x[5]:.3f}")

import csv
with open("material_pareto.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

cbar = np.array([r['cbar'] for r in rows])
J = np.array([r['J'] for r in rows])
Jmax = J.max()

# --- Figure 1 : front de Pareto ---
fig, ax = plt.subplots(figsize=(8, 5.2))
order = np.argsort(cbar)
ax.plot(cbar[order], J[order], '-', color='#888', lw=1.5, zorder=1)
sc = ax.scatter(cbar, J, c=[r['lam'] for r in rows], cmap='viridis',
                s=90, edgecolors='black', linewidths=0.6, zorder=3)
cb = fig.colorbar(sc, ax=ax, label=r'$\lambda$ (poids du coût matériau)')
# annoter le coin (lambda=0) et un point "genou"
i0 = int(np.argmax(cbar))
ax.annotate('λ=0 (coin) : J max, coût max', (cbar[i0], J[i0]),
            xytext=(-8, 9), textcoords='offset points', fontsize=9,
            ha='right', va='bottom')
# genou = design conservant >=99% de Jmax au plus faible coût
ok = np.where(J >= 0.99 * Jmax)[0]
knee = ok[np.argmin(cbar[ok])]
ax.scatter([cbar[knee]], [J[knee]], s=220, facecolors='none',
           edgecolors='#c0392b', linewidths=2.2, zorder=4)
ax.annotate(f'« genou » : {100*J[knee]/Jmax:.0f}% de Jmax\n'
            f'à coût {cbar[knee]:.2f}  (−{100*(1-cbar[knee]/cbar[i0]):.0f}% matériau)',
            (cbar[knee], J[knee]), xytext=(14, -42), textcoords='offset points',
            fontsize=9, color='#c0392b',
            arrowprops=dict(arrowstyle='->', color='#c0392b'))
ax.set_xlabel(r'Coût matériau $\bar k = \frac{1}{5}\sum_i k_i$')
ax.set_ylabel('Performance $J^\\star$')
ax.set_title('Extension — Front de Pareto : performance vs coût matériau (mesh 25)')
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig('material_pareto_front.png', dpi=150); plt.close()

# --- Figure 2 : évolution des designs ---
fig, ax = plt.subplots(figsize=(8, 5))
lam_arr = np.array([r['lam'] for r in rows])
for i, name in enumerate(['k1', 'k2', 'k3', 'k4', 'k5']):
    ax.plot(lam_arr, [r[name] for r in rows], 'o-', lw=1.8, label=name)
ax.plot(lam_arr, [r['Bi'] for r in rows], 's--', color='black', lw=1.5, label='Bi')
ax.set_xlabel(r'$\lambda$ (poids du coût matériau)')
ax.set_ylabel('Valeur optimale du paramètre')
ax.set_title('Extension — Design optimal vs poids du coût matériau')
ax.grid(alpha=0.3); ax.legend(ncol=3)
fig.tight_layout(); fig.savefig('material_design_vs_lambda.png', dpi=150); plt.close()

print(f"\nJmax (coin) = {Jmax:.4f} à coût {cbar[i0]:.3f}")
print(f"Genou : J={J[knee]:.4f} ({100*J[knee]/Jmax:.1f}% de Jmax) à coût {cbar[knee]:.3f} "
      f"(économie matériau {100*(1-cbar[knee]/cbar[i0]):.0f}%)")
print("Figures: material_pareto_front.png, material_design_vs_lambda.png")
