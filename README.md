# Optimisation – Heat Conduction (HEAT-COND)

Plateforme d'optimisation automatique pour un problème de conduction thermique 2D stationnaire (benchmark **HEAT-COND**), couplant un solveur éléments finis **FreeFEM++** à un pilote d'optimisation **Python**.

> Projet de cours — *Optimization and Numerical Analysis* (MATH6304P-260-M01), SJTU – SPEIT, Spring 2026.
> **Auteurs : Laurent ZHU, Alexandre LE DROUCPEET** — Enseignant : Prof. Helin Gong, TA : Zhipu Cui.

## Le problème

On résout $-\nabla\cdot(k\,\nabla T)=0$ sur un domaine en peigne (tronc + 5 paires d'ailettes), avec $T=1$ à la base (Dirichlet) et une condition de Robin (nombre de Biot) sur le reste de la frontière. L'objectif est de **maximiser la température moyenne $J$ sur la frontière des ailettes** par rapport au vecteur de design $(k_1,\dots,k_5,\mathrm{Bi})$, sous contraintes de bornes.

**Résultat central** : la fonctionnelle est lisse et unimodale, avec un optimum au coin du domaine admissible $x^\star=(1,1,1,1,1,\,0{,}01)$, soit $J^\star\approx0{,}730$ (maillage 50), une amélioration d'environ 850 % par rapport au design uniforme initial. Sur ce paysage, **Nelder-Mead** est la méthode la plus efficace (115 évaluations), Differential Evolution sert de vérification d'optimalité globale, et l'hybride **Adam → L-BFGS** illustre la valeur du préconditionnement adaptatif sur un problème mal conditionné. Une extension (pénalisation du coût matériau) trace le front de Pareto performance/coût : −32 % de matériau pour seulement −0,8 % de performance.

## Contenu du dépôt

```
.
├── CourseProjectIntroduction(English).pdf   # sujet du projet
├── heat_opti_final/                         # plateforme (code + études + rapport)
│   ├── scripts/            # mesh.edp, solver.edp (FreeFEM++)
│   ├── src/                # interface FreeFEM, optimiseurs, visualisation…
│   ├── app.py              # interface graphique Tkinter
│   ├── main.py             # pipeline complet en ligne de commande
│   ├── comparison_methods.ipynb   # les 7 études comparatives
│   ├── standalone_compare/ # solveur P1 NumPy (vérification, sans FreeFEM)
│   ├── results_compare/    # figures + données des études (sorties du notebook)
│   ├── final_report.md     # rapport final (source Markdown)
│   └── report_overleaf/    # rapport final LaTeX : main.tex + figures + main.pdf
├── archive/                # anciens rapports intermédiaires (historique)
└── README.md
```

**Le rapport final à rendre est [`heat_opti_final/report_overleaf/main.pdf`](heat_opti_final/report_overleaf/main.pdf)** (source : `main.tex`, compilable telle quelle sur Overleaf avec le dossier `figures/`).

## Démarrage rapide

```bash
cd heat_opti_final
cp .env.example .env        # renseigner FREEFEM_PATH (chemin vers FreeFem++)
pip install -r requirements.txt
python app.py               # interface graphique (recommandé)
```

Voir [`heat_opti_final/README.md`](heat_opti_final/README.md) pour le détail (modes d'optimisation, choix d'algorithme, sorties, reproduction des études). La comparaison des optimiseurs est reproductible **sans installation de FreeFEM** via `standalone_compare/compare_opt.py` (solveur P1 NumPy validé contre FreeFEM à 6 chiffres significatifs).

## Prérequis

- Python ≥ 3.10 (numpy, scipy, pandas, matplotlib, tqdm, python-dotenv ; tkinter inclus avec Python)
- FreeFEM++ ≥ 4.13 — https://freefem.org (optionnel pour `standalone_compare/`)
