"""Figure d'évolution de l'objectif pour la méthode retenue (Partie II).

Trace, pour la méthode retenue du modèle paramétrique, la
courbe best-so-far de l'objectif F (et de la performance Q) :
  - à gauche : en fonction du nombre d'évaluations PDE (efficacité d'échantillonnage) ;
  - à droite : en fonction du temps CPU cumulé (efficacité en temps de calcul).

Les données proviennent du cache d'historiques `results/part2/methods_param.json`
produit par le notebook `notebooks/part2_material_geometry.ipynb` — aucun appel à
FreeFEM n'est nécessaire pour régénérer la figure.

    python scripts/plot_objective_evolution.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heatcond import config

CACHE = config.RESULTS_PART2 / "methods_param.json"
OUT = config.RESULTS_PART2 / "part2_objective_evolution.png"


def main():
    if not CACHE.exists():
        raise SystemExit(
            f"Cache introuvable : {CACHE}\n"
            "Exécutez d'abord le notebook part2_material_geometry.ipynb."
        )
    d = json.load(open(CACHE))
    best_name = d.get("meta", {}).get("best_method", "Differential Evolution")
    de = d["study1"][best_name]
    hist = de["history"]
    curve = np.array(de["curve"])                 # best-so-far F
    cum_time = np.cumsum([h["time"] for h in hist])
    n = np.arange(1, len(curve) + 1)

    # best-so-far Q (performance) du meilleur design au fil des évaluations
    bestF = -np.inf
    bQ = np.nan
    runQ = []
    for h in hist:
        if h["F"] > bestF:
            bestF, bQ = h["F"], h["Q"]
        runQ.append(bQ)
    runQ = np.array(runQ)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    ax[0].plot(n, curve, color="C0", lw=2.2, label=r"objectif $F$ (best-so-far)")
    ax[0].plot(n, runQ, color="C2", lw=1.6, ls="--", label=r"$Q$ (chaleur dissipée)")
    ax[0].set_xlabel("Évaluations PDE")
    ax[0].set_ylabel("Valeur (best-so-far)")
    ax[0].set_title(f"Évolution de l'objectif vs évaluations — {best_name}")
    ax[0].legend(loc="lower right", fontsize=9)
    ax[0].grid(alpha=0.3)

    ax[1].plot(cum_time, curve, color="C3", lw=2.2, label=r"objectif $F$ (best-so-far)")
    ax[1].plot(cum_time, runQ, color="C2", lw=1.6, ls="--", label=r"$Q$ (chaleur dissipée)")
    ax[1].set_xlabel("Temps CPU cumulé (s)")
    ax[1].set_ylabel("Valeur (best-so-far)")
    ax[1].set_title(f"Évolution de l'objectif vs temps CPU — {best_name}")
    ax[1].legend(loc="lower right", fontsize=9)
    ax[1].grid(alpha=0.3)

    plt.suptitle(
        f"{best_name} (méthode retenue) — convergence "
        f"[{len(n)} évals, {cum_time[-1]:.0f} s, $F^\\star$={curve[-1]:.4f}]",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    config.ensure_dirs()
    plt.savefig(OUT, dpi=140)
    print(f"Figure écrite : {OUT}")


if __name__ == "__main__":
    main()
