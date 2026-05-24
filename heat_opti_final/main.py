import json
import pandas as pd
from src.optimization import (
    run_differential_evolution, run_nelder_mead, run_basinhopping
)
from src.visualization import plot_convergence
from src.utils import save_history, load_best_design
from src.freefem_interface import write_params, run_freefem, read_objective

def evaluate_initial_design(mesh_size=50):
    x0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    write_params(x0)
    run_freefem(doplot=0, mesh_size=mesh_size) 
    J0 = read_objective()
    print("\n" + "="*50)
    print("CONCEPTION INITIALE")
    print(f"J initial = {J0:.8f}")
    print("="*50 + "\n")

def plot_final_best():
    best_x = load_best_design()
    if best_x is None:
        print("Aucun meilleur design trouvé. L'optimisation n'a peut-être pas été lancée.")
        return
    print("\n--- Génération de la figure du meilleur design ---")
    write_params(best_x)
    run_freefem("-doplot", "1")
    print("Figure affichée. Fermez la fenêtre pour continuer.")

def main():
    with open("config/opt_config.json", "r") as f:
        config = json.load(f)
    bounds = config["bounds"]
    
    evaluate_initial_design()
    
    methods = [
        ("Differential Evolution", run_differential_evolution, config["optimization"]["differential_evolution"]),
        ("Nelder-Mead", run_nelder_mead, config["optimization"]["nelder_mead"]),
        ("Basinhopping", run_basinhopping, config["optimization"]["basinhopping"])
    ]
    
    all_results = []
    for name, func, kwargs in methods:
        print(f"\n--- Lancement de {name} ---\n")
        try:
            res = func(bounds, **kwargs)
            all_results.append(res)
            print(f"  Terminé en {res['time']:.2f} s, meilleur J = {res['best_J']:.8f}, évaluations = {res['n_eval']}")
        except Exception as e:
            print(f"  Erreur : {e}")
    
    if all_results:
        df_summary = pd.DataFrame(all_results)
        df_summary = df_summary[['method', 'best_J', 'n_eval', 'time', 'success']]
        df_summary.columns = ['Méthode', 'Meilleur J', 'Nb évaluations', 'Temps (s)', 'Convergence']
        df_summary = df_summary.sort_values('Meilleur J', ascending=False)
        print("\n" + "="*70)
        print("TABLEAU DE SYNTHÈSE DES MÉTHODES")
        print("="*70)
        print(df_summary.to_string(index=False))
        df_summary.to_csv("results/method_comparison.csv", index=False)
        
        from src.optimization import history
        save_history(history)
        plot_convergence(history)
        
        plot_final_best()

if __name__ == "__main__":
    main()