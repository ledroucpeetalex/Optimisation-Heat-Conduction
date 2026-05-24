import numpy as np
from .optimization import run_nelder_mead, run_basinhopping, reset_optimization
from .freefem_interface import run_freefem, write_params, read_objective

def sensitivity_to_initial_point(bounds, n_trials=5, method="nelder_mead", maxiter=30):
    results = []
    for i in range(n_trials):
        x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        print(f"\nEssai {i+1}/{n_trials} - point initial : {x0.round(4)}")
        if method == "nelder_mead":
            res = run_nelder_mead(bounds, x0=x0, maxiter=maxiter)
        elif method == "basinhopping":
            res = run_basinhopping(bounds, x0=x0, niter=maxiter//2)
        else:
            raise ValueError("Méthode non supportée")
        results.append(res["best_J"])
        reset_optimization()
    print(f"\n--- Résumé sensibilité ({method}) ---")
    print(f"Moyenne J = {np.mean(results):.6f} ± {np.std(results):.6f}")
    print(f"Min = {np.min(results):.6f}, Max = {np.max(results):.6f}")
    return results

def compare_mesh_sizes(bounds, mesh_sizes=[25, 50, 100], method="nelder_mead", maxiter=30):
    results = {}
    import src.optimization as opt
    original_evaluate = opt.evaluate

    for ms in mesh_sizes:
        print(f"\n--- Maillage {ms}x{ms} ---")
        def evaluate_with_mesh(x, mesh_size=None):
            write_params(x)
            run_freefem(doplot=0, mesh_size=ms)
            return -read_objective()
        opt.evaluate = evaluate_with_mesh
        try:
            if method == "nelder_mead":
                res = run_nelder_mead(bounds, maxiter=maxiter, mesh_size=ms)
            else:
                res = run_basinhopping(bounds, niter=maxiter//2, mesh_size=ms)
        finally:
            opt.evaluate = original_evaluate
        results[ms] = res["best_J"]
        reset_optimization()
    print("\n--- Comparaison des maillages ---")
    for ms, J in results.items():
        print(f"  {ms}x{ms} -> J = {J:.6f}")
    return results