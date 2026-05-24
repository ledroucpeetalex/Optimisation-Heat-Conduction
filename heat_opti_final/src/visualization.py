import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_convergence(history):
    df = pd.DataFrame(history)
    best_values = []
    current_best = -float('inf')
    for value in df["J"]:
        current_best = max(current_best, value)
        best_values.append(current_best)
    plt.figure(figsize=(10,6))
    plt.plot(best_values)
    plt.xlabel("Évaluation")
    plt.ylabel("Meilleur objectif J")
    plt.title("Convergence de l'optimisation")
    plt.grid(True)
    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", "convergence.png"), dpi=300)
    plt.show()
    plt.close()