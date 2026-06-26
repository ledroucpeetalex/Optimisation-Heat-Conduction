"""Optimiseurs HEAT-COND, séparés par modèle.

* :mod:`heatcond.optimizers.simple` — modèle simple (objectif J, 6 variables
  continues) : Nelder-Mead, Differential Evolution, L-BFGS-B, Adam, hybride
  Adam→L-BFGS, multi-start, basinhopping ;
* :mod:`heatcond.optimizers.parametric` — modèle paramétrique (objectif F,
  matériaux discrets + géométrie continue) : **les trois mêmes méthodes que la
  Partie I** (Nelder-Mead, Differential Evolution, Adam→L-BFGS), plus le
  raffinement local et le pipeline multi-fidélité ;
* :mod:`heatcond.optimizers.algorithms` — boîte à outils générique des optimiseurs :
  enveloppes **scipy.optimize** (NM, DE, L-BFGS-B), Adam custom, et optimisation
  **bayésienne** (scikit-optimize). Réutilisée par les deux modèles.
"""
