"""Programme principal en ligne de commande.

- Évalue le design initial.
- Lance les 3 méthodes (DE, Nelder-Mead, Adam → L-BFGS).
- Génère convergence par méthode + convergence comparée.
- Génère les champs T initial vs optimisé.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # racine du dépôt pour `import heatcond`

import json
from pathlib import Path

import pandas as pd

from heatcond.freefem import ensure_mesh, run_solver
from heatcond.optimizers.simple import (
    run_adam_then_lbfgs,
    run_differential_evolution,
    run_nelder_mead,
)
from heatcond.utils import RESULTS_DIR, load_best_design, save_history
from heatcond.visualization import (
    plot_convergence,
    plot_convergence_comparison,
    plot_temperature_comparison,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "opt_config.json"


def evaluate_initial_design(mesh_size: int = 50, t_out: Path | None = None) -> float:
    x0 = [0.5] * 5 + [0.5]
    J0 = run_solver(x0, mesh_size=mesh_size, doplot=0, t_out=t_out)
    print("\n" + "=" * 50)
    print("CONCEPTION INITIALE")
    print(f"J initial (x = [0.5,...,0.5]) = {J0:.8f}")
    print("=" * 50 + "\n")
    return J0


def main() -> None:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    bounds = config["bounds"]
    opt_cfg = config["optimization"]
    mesh_size = opt_cfg["differential_evolution"]["mesh_size"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_mesh(mesh_size)  # cache à l'avance

    # 1. Design initial + champ T initial sauvegardé
    T_init_file = RESULTS_DIR / "T_initial.dat"
    evaluate_initial_design(mesh_size=mesh_size, t_out=T_init_file)

    # 2. Optimisation par 3 méthodes
    methods = [
        ("Differential Evolution", run_differential_evolution, opt_cfg["differential_evolution"]),
        ("Nelder-Mead",            run_nelder_mead,            opt_cfg["nelder_mead"]),
        ("Adam-then-LBFGS",        run_adam_then_lbfgs,        opt_cfg["adam_then_lbfgs"]),
    ]

    all_results = []
    histories: dict[str, list] = {}
    for name, func, kwargs in methods:
        print(f"\n--- Lancement de {name} ---\n")
        try:
            res = func(bounds, **kwargs)
            all_results.append(res)
            histories[name] = res["history"]
            # Convergence individuelle
            plot_convergence(res["history"], filename=f"convergence_{name.replace(' ', '_')}.png")
            print(f"  Fini en {res['time']:.1f}s | J* = {res['best_J']:.8f} | n_eval = {res['n_eval']}")
        except Exception as e:
            print(f"  Erreur : {e}")

    if not all_results:
        return

    # 3. Tableau de synthèse
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in all_results])
    df = df[["method", "best_J", "n_eval", "time", "success"]]
    df.columns = ["Méthode", "Meilleur J", "Nb évaluations", "Temps (s)", "Convergence"]
    df = df.sort_values("Meilleur J", ascending=False)
    print("\n" + "=" * 70)
    print("TABLEAU DE SYNTHÈSE DES MÉTHODES")
    print("=" * 70)
    print(df.to_string(index=False))
    df.to_csv(RESULTS_DIR / "method_comparison.csv", index=False)

    # 4. Convergence comparée + historique combiné
    plot_convergence_comparison(histories)
    # Historique du best run (celui avec J max)
    best_run = max(all_results, key=lambda r: r["best_J"])
    save_history(best_run["history"])

    # 5. Champ T optimisé + comparaison avec champ initial
    best_x = load_best_design()
    if best_x is not None:
        T_opt_file = RESULTS_DIR / "T_optimized.dat"
        J_opt = run_solver(best_x, mesh_size=mesh_size, doplot=0, t_out=T_opt_file)
        print(f"\nJ optimisé (recalculé) = {J_opt:.8f}")
        mesh_path = ensure_mesh(mesh_size)
        plot_temperature_comparison(T_init_file, T_opt_file, mesh_path=mesh_path)
        print("Comparaison T initial / T optimisé sauvegardée dans results/T_comparison.png")


if __name__ == "__main__":
    main()
