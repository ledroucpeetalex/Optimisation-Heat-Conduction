"""Étude de sensibilité : point initial, paramètre par paramètre, maillage."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # racine du dépôt pour `import heatcond`

import json
from pathlib import Path

from heatcond.meta_optimization import grid_search_de
from heatcond.sensitivity import (
    compare_mesh_sizes,
    parameter_sweep,
    sensitivity_to_initial_point,
)
from heatcond.utils import load_best_design
from heatcond.visualization import plot_parameter_sweep

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "opt_config.json"


def main() -> None:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    bounds = config["bounds"]

    print("\n" + "=" * 60)
    print("1) SENSIBILITÉ AU POINT INITIAL (Nelder-Mead)")
    print("=" * 60)
    sensitivity_to_initial_point(bounds, n_trials=5, method="nelder_mead", maxiter=30)

    print("\n" + "=" * 60)
    print("2) SWEEP PARAMÈTRE PAR PARAMÈTRE")
    print("=" * 60)
    x_ref = load_best_design() or [0.5] * 5 + [0.5]
    print(f"Point de référence : {x_ref}")
    sweep = parameter_sweep(x_ref, bounds, n_points=11, mesh_size=50)
    plot_parameter_sweep(sweep)

    print("\n" + "=" * 60)
    print("3) COMPARAISON DES MAILLAGES")
    print("=" * 60)
    compare_mesh_sizes(bounds, mesh_sizes=[25, 50, 100], method="nelder_mead", maxiter=20)

    print("\n" + "=" * 60)
    print("4) META-OPTIMISATION DES HYPERPARAMETRES DE DE")
    print("=" * 60)
    grid_search_de(bounds, max_evals_per_config=40, mesh_size=25)


if __name__ == "__main__":
    main()
