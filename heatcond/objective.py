"""Objectif du modèle paramétrique : évalue F(x) via FreeFEM et journalise tout.

``ParametricObjective`` encapsule un appel solveur (FreeFEM ou solveur NumPy de référence) (performance Q, plus coût et
masse calculés depuis le catalogue de matériaux) et enregistre l'historique
complet (Q, J, coût, masse, F, temps par évaluation) ainsi que la courbe
best-so-far. Les optimiseurs de :mod:`heatcond.optimizers.parametric` partagent
cet objet.
"""

import time
from typing import Optional

import numpy as np

from .materials import unpack, k_vector, cost_and_mass


class ParametricObjective:
    """Évalue F(x) = Q - lam_cost·coût - lam_mass·masse et journalise.

    ``mesh_size`` fixe la fidélité (ex. 20 pour la recherche, 50 pour le raffinement).
    """

    def __init__(self, lam_cost: float = 0.0, lam_mass: float = 0.0,
                 mass_budget: Optional[float] = None, mesh_size: int = 25,
                 verbose: bool = False, solver=None):
        self.lam_cost = lam_cost
        self.lam_mass = lam_mass
        self.mass_budget = mass_budget
        self.mesh_size = mesh_size
        self.verbose = verbose
        # Solveur (Q, J) = f(k, t, l, Bi, mesh_size). Par défaut FreeFEM ;
        # on peut injecter heatcond.reference_solver_param.solve_param pour
        # rejouer les études sans FreeFEM.
        if solver is None:
            from .freefem import run_solver_param
            solver = run_solver_param
        self.solver = solver
        self.reset()

    def reset(self):
        self.history = []
        self.curve = []          # best-so-far F après chaque évaluation
        self.n = 0
        self.best = -np.inf
        self.best_x = None
        self.best_info = None

    def F(self, x):
        m, t, l, Bi = unpack(x)
        k = k_vector(m)
        t0 = time.time()
        Q, J = self.solver(k, t, l, Bi, mesh_size=self.mesh_size)
        dt = time.time() - t0
        cost, mass = cost_and_mass(m, t, l)

        # Performance = Q (chaleur dissipée). J (température moyenne) gardé pour info.
        F = Q - self.lam_cost * cost - self.lam_mass * mass
        if self.mass_budget is not None and mass > self.mass_budget:
            F -= 10.0 * (mass - self.mass_budget)   # pénalité de dépassement

        rec = dict(n=self.n, Q=Q, J=J, cost=cost, mass=mass, F=F, time=dt,
                   m=m.tolist(), t=list(map(float, t)), l=list(map(float, l)),
                   Bi=float(Bi))
        self.history.append(rec)
        if F > self.best:
            self.best = F
            self.best_x = np.asarray(x, dtype=float).copy()
            self.best_info = rec
        self.curve.append(self.best)
        if self.verbose:
            print(f"  éval {self.n:3d} | Q={Q:.4f} J={J:.4f} coût={cost:.3f} "
                  f"masse={mass:.3f} F={F:.4f} ({dt:.2f}s)")
        self.n += 1
        return F

    def negF(self, x):
        return -self.F(x)


class SimpleObjective:
    """Objectif du modèle simple (Partie I) : maximise la température moyenne J
    pour le design ``x = [k1..k5, Bi]`` à géométrie fixe.

    Utilisé par les études NumPy de la Partie I (reproductibles sans FreeFEM) ;
    le solveur est interchangeable et a la même signature ``(k, t, l, Bi, mesh_size)``
    que :func:`heatcond.freefem.run_solver_param`.
    """

    def __init__(self, mesh_size: int = 25, solver=None, t: float = 0.06, l: float = 0.45):
        self.mesh_size = mesh_size
        self.t = [t] * 5
        self.l = [l] * 5
        if solver is None:
            from .reference_solver_param import solve_param
            solver = solve_param
        self.solver = solver
        self.reset()

    def reset(self):
        self.history = []
        self.curve = []
        self.n = 0
        self.best = -np.inf
        self.best_x = None
        self.best_info = None

    def J(self, x):
        import time
        x = np.asarray(x, dtype=float)
        k = x[:5]
        Bi = float(x[5])
        t0 = time.time()
        Q, J = self.solver(k, self.t, self.l, Bi, mesh_size=self.mesh_size)
        dt = time.time() - t0
        rec = dict(n=self.n, J=J, Q=Q, time=dt, k=k.tolist(), Bi=Bi)
        self.history.append(rec)
        if J > self.best:
            self.best = J
            self.best_x = x.copy()
            self.best_info = rec
        self.curve.append(self.best)
        self.n += 1
        return J

    def negJ(self, x):
        return -self.J(x)
