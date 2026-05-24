import time
import numpy as np
from tqdm import tqdm
from scipy.optimize import differential_evolution, minimize, basinhopping
from .freefem_interface import write_params, run_freefem, read_objective
from .utils import save_best_design

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

def evaluate(x, mesh_size=50):
    global iteration_counter, best_J, best_x, history
    start = time.time()
    write_params(x)
    run_freefem(doplot=0, mesh_size=mesh_size)
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

def run_differential_evolution(bounds, maxiter=10, popsize=5, mesh_size=50):
    reset_optimization()
    pbar = tqdm(total=maxiter, desc="Differential Evolution", unit="gen")
    def callback(xk, conv):
        pbar.update(1)
        return False
    start = time.time()
    # On ne passe PAS disp du tout (valeur par défaut = False)
    result = differential_evolution(
        lambda x: evaluate(x, mesh_size=mesh_size),
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        callback=callback
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

def run_nelder_mead(bounds, x0=None, maxiter=100, mesh_size=50):
    reset_optimization()
    if x0 is None:
        x0 = [(b[0]+b[1])/2 for b in bounds]
    pbar = tqdm(total=maxiter, desc="Nelder-Mead", unit="iter")
    def callback(xk):
        pbar.update(1)
        return False
    start = time.time()
    result = minimize(
        lambda x: evaluate(x, mesh_size=mesh_size),
        x0,
        method='Nelder-Mead',
        bounds=bounds,
        callback=callback,
        options={'maxiter': maxiter, 'disp': False}
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

def run_basinhopping(bounds, x0=None, niter=100, mesh_size=50):
    reset_optimization()
    if x0 is None:
        x0 = [(b[0]+b[1])/2 for b in bounds]
    pbar = tqdm(total=niter, desc="Basinhopping", unit="iter")
    class CallbackBH:
        def __init__(self):
            self.i = 0
        def __call__(self, x, f, accepted):
            self.i += 1
            pbar.update(1)
    start = time.time()
    result = basinhopping(
        lambda x: evaluate(x, mesh_size=mesh_size),
        x0,
        niter=niter,
        minimizer_kwargs={'bounds': bounds},
        callback=CallbackBH()
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