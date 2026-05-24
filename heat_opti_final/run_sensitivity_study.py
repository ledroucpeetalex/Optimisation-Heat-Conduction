import json
from src.sensitivity import sensitivity_to_initial_point, compare_mesh_sizes
from src.meta_optimization import grid_search_de

def main():
    bounds = [(0.1, 1.0), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0), (0.01, 1.0)]
    
    print("\n" + "="*60)
    print("ÉTUDE DE SENSIBILITÉ AU POINT INITIAL (Nelder-Mead)")
    print("="*60)
    sensitivity_to_initial_point(bounds, n_trials=5, method="nelder_mead", maxiter=30)
    
    print("\n" + "="*60)
    print("COMPARAISON DES MAILLAGES")
    print("="*60)
    compare_mesh_sizes(bounds, mesh_sizes=[25, 50, 100], method="nelder_mead", maxiter=20)
    
    print("\n" + "="*60)
    print("MÉTA-OPTIMISATION DES HYPERPARAMÈTRES DE DE")
    print("="*60)
    grid_search_de(bounds, max_evals_per_config=40)

if __name__ == "__main__":
    main()