import numpy as np
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import os

history = []

# comment

def evaluate(x):

    np.savetxt("params.txt", x)

    subprocess.run(
        ["FreeFem++", "heat_solver.edp"],
        check=True
    )

    with open("objective.txt","r") as f:
        J = float(f.read())

    print("J =", J)

    history.append([*x, J])

    return -J

bounds = [

    (0.1,1.0),
    (0.1,1.0),
    (0.1,1.0),
    (0.1,1.0),
    (0.1,1.0),
    (0.01,1.0)
]

result = differential_evolution(
    evaluate,
    bounds,
    maxiter=10,
    popsize=5,
    disp=True
)

best_x = result.x
best_J = -result.fun

print("\nBEST PARAMETERS")
print(best_x)

print("\nBEST OBJECTIVE")
print(best_J)

os.makedirs("results", exist_ok=True)

df = pd.DataFrame(
    history,
    columns=[
        "k1",
        "k2",
        "k3",
        "k4",
        "k5",
        "Bi",
        "J"
    ]
)

df.to_csv(
    "results/history.csv",
    index=False
)

plt.plot(df["J"])
plt.xlabel("Evaluation")
plt.ylabel("Objective J")
plt.grid(True)

plt.savefig("results/convergence.png")

plt.show()