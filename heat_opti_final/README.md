# Heat-Cond Optimization Platform v4

## Installation

1. Copier et renseigner `FREEFEM_PATH` (chemin vers FreeFem++.exe)
2. Installer les dépendances : `pip install -r requirements.txt`
3. Vérifier que `scripts/heat_solver.edp` est présent

## Utilisation – quel script lancer ?

| Ce que vous voulez faire                                       | Commande                          |
|----------------------------------------------------------------|-----------------------------------|
| Calcul ponctuel (saisir k1..k5, Bi et obtenir J)               | `python user_interface.py`        |
| Optimisation automatique (3 méthodes, comparaison, graphiques) | `python main.py`                  |
| Études de sensibilité + méta-optimisation                      | `python run_sensitivity_study.py` |
| Regénérer la figure du meilleur design                         | `python plot_best.py`             |

Tous les résultats sont sauvegardés dans le dossier `results/`.

Auteurs : Laurent ZHU, Alexandre LE DROUCPEET – SJTU SPEIT