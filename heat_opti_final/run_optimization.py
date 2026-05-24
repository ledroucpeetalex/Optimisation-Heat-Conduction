import os
import subprocess
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize, basinhopping
from tqdm import tqdm
from dotenv import load_dotenv

# ============================================================
# 1. CHARGEMENT DES CHEMINS
# ============================================================
load_dotenv()
FREEFEM_EXEC = os.getenv("FREEFEM_PATH")
if FREEFEM_EXEC is None:
    raise ValueError("FREEFEM_PATH non défini dans .env")

FREEFEM_SCRIPT = "scripts/heat_solver.edp"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 2. FONCTIONS D'INTERFACE AVEC FreeFEM
# ============================================================
def write_params(x, filename="params.txt"):
    with open(filename, "w") as f:
        for val in x:
            f.write(f"{val}\n")

def run_freefem(mesh_size=50, doplot=0):
    cmd = [FREEFEM_EXEC, FREEFEM_SCRIPT, "-meshsize", str(mesh_size), "-doplot", str(doplot)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError("FreeFEM execution failed")
    return result

def read_objective(filename="objective.txt"):
    with open(filename, "r") as f:
        return float(f.readline().strip())

# ============================================================
# 3. GESTION DE L'HISTORIQUE ET MEILLEUR DESIGN
# ============================================================
history = []
iteration_counter = 0
best_J = -np.inf
best_x = None

def reset_optimization():
    global history, iteration_counter, best_J, best_x
    history = []
    iteration_counter = 0
    best_J = -np.inf
    best_x = None

def save_best_design(best_x, best_J):
    # Fichier texte lisible
    filepath_txt = os.path.join(RESULTS_DIR, "best_design.txt")
    with open(filepath_txt, "w") as f:
        f.write("===== BEST DESIGN =====\n\n")
        f.write(f"k1 = {best_x[0]}\n")
        f.write(f"k2 = {best_x[1]}\n")
        f.write(f"k3 = {best_x[2]}\n")
        f.write(f"k4 = {best_x[3]}\n")
        f.write(f"k5 = {best_x[4]}\n")
        f.write(f"Bi = {best_x[5]}\n\n")
        f.write(f"Best objective J = {best_J}\n")
    # Fichier numérique pour FreeFEM (plot final)
    with open("best_params.txt", "w") as f:
        for val in best_x:
            f.write(f"{val}\n")

def save_history():
    if history:
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(RESULTS_DIR, "optimization_history.csv"), index=False)

# ============================================================
# 4. FONCTION D'ÉVALUATION (pour l'optimiseur)
# ============================================================
def evaluate(x, mesh_size=50):
    global iteration_counter, best_J, best_x, history
    start = time.time()
    write_params(x)
    run_freefem(mesh_size=mesh_size, doplot=0)
    J = read_objective()
    elapsed = time.time() - start

    row = {
        "iteration": iteration_counter,
        "k1": x[0], "k2": x[1], "k3": x[2],
        "k4": x[3], "k5": x[4], "Bi": x[5],
        "J": J, "time": elapsed, "mesh_size": mesh_size
    }
    history.append(row)

    if J > best_J:
        best_J = J
        best_x = np.copy(x)
        save_best_design(best_x, best_J)

    print(f"  Éval {iteration_counter:3d} | J = {J:.8f} | temps = {elapsed:.2f}s")
    iteration_counter += 1
    return -J

# ============================================================
# 5. MÉTHODES D'OPTIMISATION
# ============================================================
def run_differential_evolution(bounds, maxiter=10, popsize=5, mesh_size=50, **kwargs):
    reset_optimization()
    pbar = tqdm(total=maxiter, desc="Differential Evolution", unit="gen")
    def callback(xk, conv):
        pbar.update(1)
        return False
    start = time.time()
    result = differential_evolution(
        lambda x: evaluate(x, mesh_size=mesh_size), bounds,
        maxiter=maxiter, popsize=popsize, callback=callback, disp=False, **kwargs
    )
    pbar.close()
    elapsed = time.time() - start
    return {
        'method': 'Differential Evolution',
        'best_J': -result.fun,
        'best_x': result.x,
        'n_eval': len(history),
        'time': elapsed,
        'success': result.success,
        'message': result.message
    }

def run_nelder_mead(bounds, x0=None, maxiter=100, mesh_size=50, **kwargs):
    reset_optimization()
    if x0 is None:
        x0 = [(b[0]+b[1])/2 for b in bounds]
    pbar = tqdm(total=maxiter, desc="Nelder-Mead", unit="iter")
    def callback(xk):
        pbar.update(1)
        return False
    start = time.time()
    result = minimize(
        lambda x: evaluate(x, mesh_size=mesh_size), x0, method='Nelder-Mead',
        bounds=bounds, callback=callback, options={'maxiter': maxiter, 'disp': False}, **kwargs
    )
    pbar.close()
    elapsed = time.time() - start
    return {
        'method': 'Nelder-Mead',
        'best_J': -result.fun,
        'best_x': result.x,
        'n_eval': len(history),
        'time': elapsed,
        'success': result.success,
        'message': result.message
    }

def run_basinhopping(bounds, x0=None, niter=100, mesh_size=50, **kwargs):
    reset_optimization()
    if x0 is None:
        x0 = [(b[0]+b[1])/2 for b in bounds]
    pbar = tqdm(total=niter, desc="Basinhopping", unit="iter")
    class CallbackBH:
        def __init__(self): self.i = 0
        def __call__(self, x, f, accepted):
            self.i += 1
            pbar.update(1)
    start = time.time()
    result = basinhopping(
        lambda x: evaluate(x, mesh_size=mesh_size), x0, niter=niter,
        minimizer_kwargs={'bounds': bounds}, callback=CallbackBH(), **kwargs
    )
    pbar.close()
    elapsed = time.time() - start
    return {
        'method': 'Basinhopping',
        'best_J': -result.fun,
        'best_x': result.x,
        'n_eval': len(history),
        'time': elapsed,
        'success': result.lowest_optimization_result.success,
        'message': result.message
    }

# ============================================================
# 6. VISUALISATION
# ============================================================
def plot_convergence(history):
    if not history:
        print("Aucun historique à tracer.")
        return
    df = pd.DataFrame(history)
    best_values = []
    current_best = -float('inf')
    for val in df["J"]:
        current_best = max(current_best, val)
        best_values.append(current_best)
    plt.figure(figsize=(10,6))
    plt.plot(best_values)
    plt.xlabel("Évaluation")
    plt.ylabel("Meilleur objectif J")
    plt.title("Convergence de l'optimisation")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "convergence.png"), dpi=300)
    plt.show()
    plt.close()

def plot_final_best(mesh_size=50):
    if not os.path.exists("best_params.txt"):
        print("Aucun meilleur design trouvé.")
        return
    with open("best_params.txt", "r") as f:
        best_x = [float(line.strip()) for line in f.readlines()]
    print("\n--- Génération de la figure du meilleur design ---")
    write_params(best_x)
    run_freefem(mesh_size=mesh_size, doplot=1)
    print("Figure affichée. Fermez la fenêtre pour continuer.")

# ============================================================
# 7. PROGRAMME PRINCIPAL
# ============================================================
def evaluate_initial_design(mesh_size=50):
    x0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    write_params(x0)
    run_freefem(mesh_size=mesh_size, doplot=0)
    J0 = read_objective()
    print("\n" + "="*50)
    print("CONCEPTION INITIALE")
    print(f"J initial = {J0:.8f}")
    print("="*50 + "\n")

def main():
    # Paramètres par défaut (vous pouvez aussi charger depuis un JSON si vous voulez)
    bounds = [(0.1, 1.0), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0), (0.01, 1.0)]
    mesh_size = 50

    evaluate_initial_design(mesh_size)

    methods = [
        ("Differential Evolution", run_differential_evolution, {'maxiter': 5, 'popsize': 4}),
        ("Nelder-Mead", run_nelder_mead, {'maxiter': 50}),
        ("Basinhopping", run_basinhopping, {'niter': 30})
    ]

    all_results = []
    for name, func, kwargs in methods:
        print(f"\n--- Lancement de {name} (maillage {mesh_size}) ---\n")
        try:
            res = func(bounds, mesh_size=mesh_size, **kwargs)
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
        df_summary.to_csv(os.path.join(RESULTS_DIR, "method_comparison.csv"), index=False)

        save_history()
        plot_convergence(history)
        plot_final_best(mesh_size)

if __name__ == "__main__":
    main()