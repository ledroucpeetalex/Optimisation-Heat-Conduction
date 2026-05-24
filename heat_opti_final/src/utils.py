import os
import pandas as pd
import time

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_best_design(best_x, best_J):
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
    
    filepath_num = "best_params.txt"
    for attempt in range(3):
        try:
            with open(filepath_num, "w") as f:
                for val in best_x:
                    f.write(f"{val}\n")
            break
        except PermissionError:
            time.sleep(0.2)  # attendre un peu avant de réessayer

def save_history(history):
    df = pd.DataFrame(history)
    filepath = os.path.join(RESULTS_DIR, "optimization_history.csv")
    df.to_csv(filepath, index=False)

def load_best_design():
    if not os.path.exists("best_params.txt"):
        return None
    with open("best_params.txt", "r") as f:
        vals = [float(line.strip()) for line in f.readlines()]
    if len(vals) == 6:
        return vals
    return None