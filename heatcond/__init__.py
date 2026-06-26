"""HEAT-COND — plateforme d'optimisation de conduction thermique 2D (FreeFEM++ + Python).

Le package expose deux modèles cohérents du même benchmark :

* **modèle simple** (Partie I du rapport) — on optimise les conductivités
  ``(k_1..k_5, Bi)`` pour maximiser la température moyenne ``J`` sur la frontière
  des ailettes ; voir :mod:`heatcond.optimizers.simple` ;
* **modèle paramétrique** (Partie II) — chaque ailette reçoit un matériau discret
  (conductivité, prix, densité) et un dimensionnement continu (épaisseur, longueur),
  et on maximise un objectif multi-critères performance/coût/masse ; voir
  :mod:`heatcond.materials`, :mod:`heatcond.objective` et
  :mod:`heatcond.optimizers.parametric`.

Le solveur éléments finis P1 est écrit en FreeFEM++ (dossier ``freefem/``) et piloté
via :mod:`heatcond.freefem`. Une réimplémentation NumPy pure
(:mod:`heatcond.reference_solver`) permet de rejouer la comparaison des optimiseurs
de la Partie I sans installer FreeFEM.
"""

__version__ = "1.0.0"
__all__ = [
    "config",
    "freefem",
    "materials",
    "objective",
    "sensitivity",
    "reference_solver",
    "reference_solver_param",
    "visualization",
    "utils",
]
