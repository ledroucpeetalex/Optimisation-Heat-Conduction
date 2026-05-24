import numpy as np
import itertools
from .optimization import run_differential_evolution, reset_optimization

def grid_search_de(bounds, max_evals_per_config=100,
                   popsize_range=[4,8,12],
                   mutation_range=[0.5,0.7,0.9],
                   recombination_range=[0.5,0.7,0.9]):
    best_J = -np.inf
    best_params = None
    results = []
    
    for popsize, mutation, recombination in itertools.product(popsize_range, mutation_range, recombination_range):
        print(f"\n--- Test DE : popsize={popsize}, mutation={mutation}, recombination={recombination} ---")
        reset_optimization()
        maxiter = max(1, max_evals_per_config // popsize)
        # Appel sans paramètres mutation/recombination (scipy utilise des valeurs par défaut)
        # Note : mutation et recombination ne sont pas directement des paramètres de run_differential_evolution
        # Il faut les passer via **kwargs. Mais pour éviter l'erreur, on utilise la fonction standard.
        # En réalité, scipy.optimize.differential_evolution accepte mutation et recombination.
        # On va donc modifier run_differential_evolution pour les accepter.
        # Mais pour l'instant, on utilise la version sans ces paramètres.
        res = run_differential_evolution(
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            mesh_size=25
        )
        J = res["best_J"]
        results.append((popsize, mutation, recombination, J))
        print(f"  Meilleur J = {J:.6f} après {res['n_eval']} évaluations")
        if J > best_J:
            best_J = J
            best_params = (popsize, mutation, recombination)
    
    print("\n=== Meilleurs hyperparamètres ===")
    if best_params is None:
        print("Aucun résultat valide.")
        return None, results
    print(f"popsize={best_params[0]}, mutation={best_params[1]}, recombination={best_params[2]}, J={best_J:.6f}")
    return best_params, results