# Comparaison autonome des optimiseurs (sans FreeFEM)

`fem_solver.py` réimplémente en numpy pur le solveur `scripts/solver.edp`
(P1, kappa par morceaux, Robin label 2, Dirichlet label 1, J = moyenne de T
sur label 2). Validé contre FreeFEM : J(coin 1,1,1,1,1,0.01) = 0.730396 à
mesh 50, identique au J* Nelder-Mead des études FreeFEM.

`compare_opt.py` compare Adam->L-BFGS, Nelder-Mead, Differential Evolution
(+ Adam seul, L-BFGS seul) sur ce solveur. Sert à reproduire l'étude 10
sans installer FreeFEM.

    python3 compare_opt.py     # nécessite mesh_25.msh dans le dossier courant

Sorties : comparison_adam_lbfgs.csv, curves.json.

## Extension 1 — Compromis coût matériau (front de Pareto)

`pareto_material.py` maximise l'objectif scalarisé
`F_lambda(x) = J(x) - lambda * mean(k_i)` pour une grille de `lambda`, et trace
le front de Pareto **performance J vs coût matériau**. Rend le problème non
trivial (optimum intérieur, au lieu du coin). Nécessite `mesh_25.msh` dans le
dossier courant (comme `compare_opt.py`).

    python3 pareto_material.py

Sorties : `material_pareto.csv`, `material_pareto_front.png`,
`material_design_vs_lambda.png`.

Résultat clé : on peut retirer ~32 % de matériau conducteur en ne perdant que
0,8 % de performance ; et les ailettes proches de la base (k1) restent
conductrices bien plus longtemps que celles du haut (k5) quand le coût augmente.
