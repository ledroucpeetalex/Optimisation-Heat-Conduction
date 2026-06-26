"""Lancement CLI du pipeline d'optimisation du modèle enrichi.

Exécute la stratégie recommandée — Differential Evolution sur maillage grossier
puis raffinement Nelder-Mead sur maillage fin — et écrit le meilleur design.

    python scripts/run_param.py
    python scripts/run_param.py --lam-cost 0.04 --lam-mass 0.01 --maxiter 25
"""
import argparse
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # racine du dépôt pour `import heatcond`

import numpy as np

from heatcond import config
from heatcond import materials as model
from heatcond.optimizers import parametric as optimize


def main():
    p = argparse.ArgumentParser(description="Optimisation HEAT-COND enrichie")
    p.add_argument("--lam-cost", type=float, default=0.02)
    p.add_argument("--lam-mass", type=float, default=0.01)
    p.add_argument("--mass-budget", type=float, default=None)
    p.add_argument("--coarse", type=int, default=20)
    p.add_argument("--fine", type=int, default=50)
    p.add_argument("--maxiter", type=int, default=20)
    p.add_argument("--popsize", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"DE (maillage {args.coarse}) → raffinement (maillage {args.fine})")
    print(f"λ_coût={args.lam_cost}  λ_masse={args.lam_mass}  "
          f"budget_masse={args.mass_budget}\n")

    glob, fine = optimize.multifidelity(
        lam_cost=args.lam_cost, lam_mass=args.lam_mass,
        mass_budget=args.mass_budget, coarse_mesh=args.coarse,
        fine_mesh=args.fine, maxiter=args.maxiter, popsize=args.popsize,
        seed=args.seed, verbose=True,
    )

    print("\n=== Recherche globale (DE, grossier) ===")
    print(f"F={glob['best_F']:.4f}  Q={glob['best_Q']:.4f}  J={glob['best_J']:.4f}  "
          f"coût={glob['best_cost']:.3f}  masse={glob['best_mass']:.3f}  "
          f"n_eval={glob['n_eval']}  t={glob['time']:.0f}s")
    print("\n=== Raffinement (Nelder-Mead, fin) ===")
    print(f"F={fine['best_F']:.4f}  Q={fine['best_Q']:.4f}  J={fine['best_J']:.4f}  "
          f"coût={fine['best_cost']:.3f}  masse={fine['best_mass']:.3f}  "
          f"n_eval={fine['n_eval']}  t={fine['time']:.0f}s\n")
    print(model.describe(fine["best_x"]))

    config.ensure_dirs()
    out = {
        "params": vars(args),
        "best_x": np.asarray(fine["best_x"]).tolist(),
        "best_F": fine["best_F"], "best_Q": fine["best_Q"],
        "best_J": fine["best_J"],
        "best_cost": fine["best_cost"], "best_mass": fine["best_mass"],
    }
    with open(config.RESULTS_PART2 / "best_param.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nMeilleur design écrit dans results/part2/best_param.json")


if __name__ == "__main__":
    main()
