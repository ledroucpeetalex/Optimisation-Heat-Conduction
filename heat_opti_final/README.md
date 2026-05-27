# HEAT-COND Optimization Platform

Plateforme d'optimisation automatique pour le problème de conduction
thermique 2D HEAT-COND (sujet ONA, SPEIT, Spring 2026).

## Installation

1. Copier `.env.example` vers `.env` et renseigner `FREEFEM_PATH`
   (chemin absolu vers le binaire `FreeFem++`).
2. `pip install -r requirements.txt` (numpy, scipy, pandas, matplotlib,
   tqdm, python-dotenv ; `tkinter` est livré avec Python).
3. Vérifier la présence de `scripts/mesh.edp` et `scripts/solver.edp`.

## Architecture

```
heat_opti_final/
├── scripts/
│   ├── mesh.edp        # génère le maillage (1 fois par mesh_size, mis en cache)
│   └── solver.edp      # résout la PDE pour (k1..k5, Bi) passés en CLI
├── src/
│   ├── freefem_interface.py   # ensure_mesh + run_solver + parsing stdout
│   ├── optimization.py        # DE / Nelder-Mead / Basinhopping
│   ├── sensitivity.py         # point initial / sweep / mesh
│   ├── meta_optimization.py   # grid search sur popsize DE
│   ├── utils.py               # save_best_design / save_history / load_best_design
│   └── visualization.py       # convergence, comparaison T, sweep
├── config/opt_config.json     # bornes + modes + hyperparamètres
├── app.py                     # interface Tkinter
├── main.py                    # pipeline complet (3 méthodes + analyses)
├── user_interface.py          # calcul ponctuel CLI
└── run_sensitivity_study.py   # études de sensibilité
```

Les maillages générés sont mis en cache dans `cache/mesh_<size>.msh`.
Les résultats vont dans `results/` (créé automatiquement).

## Comment lancer

| Objectif                                              | Commande                          |
|-------------------------------------------------------|-----------------------------------|
| Interface graphique (recommandé)                      | `python app.py`                   |
| Calcul ponctuel CLI (saisir k1..k5, Bi)               | `python user_interface.py`        |
| Pipeline complet (3 méthodes + analyses + figures)    | `python main.py`                  |
| Études de sensibilité + méta-optimisation             | `python run_sensitivity_study.py` |

## Modes d'optimisation (interface Tkinter)

| Mode                    | mesh_size | maxiter | popsize |
|-------------------------|-----------|---------|---------|
| Recherche rapide        | 25        | 5       | 4       |
| Recherche normale       | 50        | 15      | 8       |
| Recherche approfondie   | 80        | 30      | 15      |

L'utilisateur peut aussi fournir son propre fichier `.msh` via le bouton
« Parcourir » dans la section Maillage.

## Sorties (dans `results/`)

- `best_design.txt`             — fiche lisible du meilleur design.
- `optimization_history.csv`    — historique complet des évaluations.
- `method_comparison.csv`       — tableau de synthèse des 3 méthodes.
- `convergence_<méthode>.png`   — convergence par méthode.
- `convergence_comparison.png`  — 3 méthodes sur un même graphe.
- `T_initial.dat`, `T_optimized.dat` — champs T (x y T) exportés par FreeFEM.
- `T_comparison.png`            — comparaison des champs T initial vs optimisé.
- `parameter_sweep.png`         — sensibilité one-at-a-time.

## Auteurs

Laurent ZHU, Alexandre LE DROUCPEET — SJTU SPEIT, Spring 2026.
