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
│   ├── optimization.py        # DE / NM / BH / L-BFGS / Adam→L-BFGS
│   ├── sensitivity.py         # point initial / sweep / mesh
│   ├── meta_optimization.py   # grid search sur popsize DE
│   ├── utils.py               # save_best_design / save_history / load_best_design
│   └── visualization.py       # convergence, comparaison T, sweep
├── config/opt_config.json     # bornes + modes + hyperparamètres
├── app.py                     # interface Tkinter
├── main.py                    # pipeline complet (3 méthodes + analyses)
├── user_interface.py          # calcul ponctuel CLI
├── run_sensitivity_study.py   # études de sensibilité
├── comparison_methods.ipynb   # 7 études comparatives (pour le rapport)
└── standalone_compare/        # solveur Python + comparaison (sans FreeFEM)
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

### Choix de l'algorithme (onglet Optimisation)

L'onglet **Optimisation** procède en deux étapes :

1. **① Performance maximale (sans contrainte de prix)** — maximise J seul ; donne la référence (borne haute).
2. **② Compromis performance / coût (avec contrainte de prix)** — saisir un poids λ > 0 ; maximise J − λ·k̄ et affiche la perte de performance et le matériau économisé par rapport à l'étape ①.

Un menu déroulant permet de choisir l'algorithme d'optimisation :

| Algorithme        | Type                | Remarque                                            |
|-------------------|---------------------|-----------------------------------------------------|
| **Nelder-Mead**   | local, simplexe     | **défaut recommandé** — le plus efficient ici       |
| Differential Evolution | global, populationnel | robuste, sert de vérification d'optimalité globale |
| Adam → L-BFGS     | hybride à gradient  | Adam (préconditionné) puis raffinement L-BFGS-B     |

Sur ce problème (fonctionnelle lisse, optimum sur la frontière), Nelder-Mead
atteint l'optimum global $x^\star=(1,1,1,1,1,\,0{,}01)$ au plus faible coût.
Voir le rapport (`final_report.md`) et l'étude 7 du notebook pour la comparaison
détaillée et le rôle d'Adam comme préconditionneur (L-BFGS seul stagne sur ce
problème mal conditionné).

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
