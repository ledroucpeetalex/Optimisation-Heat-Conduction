# Plateforme d'optimisation automatique d'un système 2D de conduction thermique

**HEAT-COND benchmark — FreeFEM++ et Python**

*Rapport intermédiaire*

| Cours | Optimisation et Analyse Numérique (MATH6304P-260-M01) — Spring 2026 |
|---|---|
| Établissement | Shanghai Jiao Tong University — SPEIT |
| Enseignant | Prof. Helin Gong |
| Assistant | Zhipu Cui |
| Auteurs | Laurent Zhu, Alexandre Le Droucpeet |

---

## 1. Introduction et objectif du projet

Ce rapport intermédiaire présente l'état actuel d'une plateforme logicielle développée dans le cadre du projet de cours d'Optimisation et Analyse Numérique. L'objectif scientifique est de coupler la résolution d'un problème aux dérivées partielles (PDE) elliptique de conduction thermique, posé sur une géométrie 2D non triviale, avec une boucle d'optimisation paramétrique automatisée. Le problème de référence est le benchmark HEAT-COND issu de la littérature [1].

La plateforme combine cinq éléments imposés par le cahier des charges :

- un solveur élément-fini écrit en FreeFEM++ pour la résolution de l'équation de conduction stationnaire,
- un module d'évaluation de la fonctionnelle objectif $J$ (température moyenne sur la frontière des ailettes),
- un pilote d'optimisation Python couplant FreeFEM++ à plusieurs algorithmes scipy,
- une interface graphique permettant à l'utilisateur de piloter le solveur et l'optimiseur sans ligne de commande,
- un cadre d'analyse numérique pour l'étude comparative des méthodes d'optimisation.

Le présent document décrit en détail chacun de ces éléments dans leur état actuel d'implémentation. Les résultats numériques chiffrés (tableaux de convergence, comparaisons quantitatives, etc.) seront ajoutés dans la version finale du rapport, après exécution complète du notebook d'études comparatives décrit en section 6.

---

## 2. Modèle mathématique

### 2.1. Géométrie du domaine

Le domaine de calcul $\Omega$ est inclus dans le carré unité $[0, 1]^2$. Il se compose d'un *tronc* vertical central (spine) de largeur $w_s = 0{,}10$ et de 5 paires d'ailettes (fins) horizontales d'épaisseur $t_f = 0{,}06$, positionnées aux ordonnées $y \in \{0{,}16\ ;\ 0{,}32\ ;\ 0{,}48\ ;\ 0{,}64\ ;\ 0{,}80\}$. La géométrie forme un peigne symétrique, multi-connexe par les vides entre ailettes.

Notations utilisées dans la plateforme :

- $x_g = (1 - w_s)/2 = 0{,}45$ et $x_d = x_g + w_s = 0{,}55$ : abscisses des bords intérieurs du tronc.
- $\Gamma_{\text{Base}}$ : le bord inférieur du tronc ($y = 0$, $x \in [x_g, x_d]$).
- $\Gamma_{\text{Fin}}$ : tout le reste de $\partial\Omega$ (les pointes extérieures et toutes les surfaces des ailettes).

### 2.2. Équation aux dérivées partielles et conditions aux limites

Le problème de conduction stationnaire s'écrit :

$$
\begin{cases}
-\nabla \cdot \big(k(x)\,\nabla T\big) = 0 & \text{dans } \Omega, \\
T = 1 & \text{sur } \Gamma_{\text{Base}}, \\
k(x)\,\dfrac{\partial T}{\partial n} + \mathrm{Bi}\,T = 0 & \text{sur } \Gamma_{\text{Fin}}.
\end{cases}
$$

Le coefficient $k(x)$ est par morceaux :

- $k(x) = 1$ dans le tronc central ;
- $k(x) = k_i$, $i = 1, \dots, 5$, dans l'ailette $i$ comptée du bas vers le haut.

Le nombre de Biot $\mathrm{Bi}$ pilote l'intensité du couplage convectif sur $\Gamma_{\text{Fin}}$. Les variables de design sont donc

$$
x = (k_1, k_2, k_3, k_4, k_5, \mathrm{Bi}) \in \mathbb{R}^6,
$$

avec $k_i \in [0{,}1\ ;\ 1{,}0]$ et $\mathrm{Bi} \in [0{,}01\ ;\ 1{,}0]$.

### 2.3. Fonctionnelle objectif

L'objectif est de maximiser la température moyenne sur la frontière des ailettes :

$$
J(x) \;=\; \frac{1}{|\Gamma_{\text{Fin}}|} \int_{\Gamma_{\text{Fin}}} T\,\mathrm{d}\Gamma.
$$

Physiquement, $J$ élevé signifie que le tronc parvient à transférer efficacement la chaleur ($T = 1$ en base) vers les surfaces périphériques. Le problème d'optimisation est donc $\max_x J(x)$ sous les contraintes de bornes ci-dessus.

---

## 3. Solveur élément-fini en FreeFEM++

Le solveur a été refondu pour utiliser une discrétisation propre et une interface Python sans fichier texte intermédiaire. L'implémentation est découpée en deux scripts indépendants.

### 3.1. Génération du maillage (`scripts/mesh.edp`)

La version initiale utilisait l'approche `square(meshsize, meshsize)` suivie d'un `trunc(material > 0.5)`. Cette approche pose deux problèmes :

1. les frontières du domaine sont en escalier diagonal car la triangulation ne s'aligne pas sur les bords horizontaux/verticaux de la géométrie ;
2. les arêtes intérieures créées par `trunc` reçoivent un label par défaut, ce qui entraîne une mauvaise application des conditions aux limites de Robin sur les flancs des ailettes.

Nous avons réécrit `mesh.edp` à l'aide de `buildmesh()` avec une description explicite des 44 segments de bord, parcourus dans le sens trigonométrique. Les segments sont définis par deux tableaux $(b_x, b_y)$ de 45 points et un tableau `lbl` des labels de chaque segment. Une définition unique de `border` indexée par `i` parcourt ces tableaux :

```cpp
border bd(t = 0, 1; i) {
    x = b_x[i] + t * (b_x[i+1] - b_x[i]);
    y = b_y[i] + t * (b_y[i+1] - b_y[i]);
    label = lbl[i];
}
mesh Th = buildmesh(bd(nn));
```

Le nombre de subdivisions par segment `nn[i]` est calculé proportionnellement à la longueur euclidienne du segment, multipliée par la densité globale `meshsize`. Le maillage résultant a des frontières exactement alignées sur la géométrie et une triangulation de Delaunay propre à l'intérieur.

**Schéma de labels retenu** :

- **Label 1** : la base, soit le segment unique $[x_g, x_d] \times \{0\}$. Condition de Dirichlet $T = 1$.
- **Label 2** : tout le reste de la frontière (sommet du tronc, pointes extérieures des ailettes, et l'ensemble des flancs horizontaux et verticaux des ailettes et du tronc). Condition de Robin.

Le maillage est sauvegardé dans `cache/mesh_<size>.msh`, ce qui permet sa mise en cache et sa réutilisation sans régénération.

### 3.2. Solveur PDE (`scripts/solver.edp`)

`solver.edp` prend tous ses paramètres via la ligne de commande grâce à `getARGV` : `-meshfile`, `-k1` à `-k5`, `-Bi`, `-doplot` et un optionnel `-Tout` pour exporter le champ $T$ en sortie.

La formulation variationnelle assemblée est, pour $T \in V_h$ (P1 sur $T_h$) et $v$ fonction-test :

$$
a(T, v) = \int_\Omega \kappa \, \nabla T \cdot \nabla v \,\mathrm{d}\Omega + \int_{\Gamma_{\text{Fin}}} \mathrm{Bi}\,T\,v\,\mathrm{d}\Gamma,
$$

avec $T = 1$ imposé fortement sur $\Gamma_{\text{Base}}$.

La conductivité $\kappa$ est interpolée dans $V_h$ sous forme d'une fonction indicatrice :

```cpp
Vh kappa = inSpine(x, y) ? 1.0 : (
           inFin(x, y, fy1) ? k1 : (
           inFin(x, y, fy2) ? k2 : (
           inFin(x, y, fy3) ? k3 : (
           inFin(x, y, fy4) ? k4 : (
           inFin(x, y, fy5) ? k5 : 1.0 )))));
```

L'objectif $J$ est calculé via `int1d` sur le label 2 :

```cpp
real L = int1d(Th, 2)(1.0);
real J = int1d(Th, 2)(T) / L;
cout << "Objective_J=" << J << endl;
```

La sortie standard est ainsi parsable par expression régulière côté Python, ce qui supprime la nécessité d'un fichier `objective.txt` comme dans la version initiale.

---

## 4. Module d'optimisation Python

### 4.1. Couche d'interface FreeFEM (`src/freefem_interface.py`)

Le module expose deux fonctions principales :

- `ensure_mesh(mesh_size, force=False)` : génère le maillage si absent du cache, renvoie son chemin. La mise en cache économise un appel FreeFEM par évaluation.
- `run_solver(x, mesh_size, t_out=None) -> J` : appelle `solver.edp` via `subprocess`, parse la sortie standard à la recherche d'`Objective_J=...`, et renvoie $J$. L'argument `t_out`, s'il est fourni, déclenche l'export du champ $T$ sous forme d'un fichier $(x, y, T)$ par sommet.

Un parseur natif `read_freefem_mesh()` lit également les fichiers `.msh` (format ASCII : `nv nt nbe`, vertices, triangles, edges) afin de fournir à matplotlib la triangulation native pour les visualisations.

### 4.2. Algorithmes d'optimisation (`src/optimization.py`)

Trois algorithmes `scipy.optimize` sont enveloppés dans des wrappers uniformes :

| Méthode | Type | Caractéristique principale |
|---|---|---|
| Differential Evolution (DE) | Global, stochastique | Population (`popsize × 6` individus), mutation + croisement, robuste sans gradient |
| Nelder-Mead (NM) | Local, déterministe | Simplexe en dimension 6, ne nécessite pas de gradient, sensible au point initial |
| Basinhopping (BH) | Hybride global/local | Perturbations aléatoires (Metropolis) + descente locale L-BFGS-B avec bornes |

Chaque wrapper alimente un historique d'évaluation (liste de dicts contenant l'itération, les six paramètres, $J$, le temps de calcul et la taille de maillage), sauvegarde le meilleur design rencontré, et retourne un dictionnaire normalisé contenant `best_J`, `best_x`, `n_eval`, `time`, `success` et l'historique complet. Cette uniformisation simplifie le code d'analyse et de visualisation en aval.

Deux mécanismes améliorent la reproductibilité et la flexibilité :

- Le paramètre `seed=42` par défaut sur DE et BH garantit la reproductibilité bit-à-bit des runs.
- Le mécanisme `**scipy_kwargs` permet de passer en transparence n'importe quel paramètre natif de scipy (`mutation`, `recombination`, `strategy` pour DE ; `stepsize`, `T` pour BH). C'est ce qui permet les études d'hyperparamètres du notebook (cf. section 6).

Un drapeau global `VERBOSE = True/False` contrôle la verbosité des prints et des barres `tqdm`. Posé à `False` dans le notebook, il évite que la console soit polluée par plusieurs centaines de lignes d'évaluation.

---

## 5. Architecture de la plateforme

### 5.1. Organisation des fichiers

```
heat_opti_final/
├── scripts/
│   ├── mesh.edp        # génération du maillage (44 bords, buildmesh)
│   └── solver.edp      # solveur PDE paramétrisé par getARGV
├── src/
│   ├── freefem_interface.py   # ensure_mesh, run_solver, read_freefem_mesh
│   ├── optimization.py        # DE, NM, BH (wrappers scipy)
│   ├── visualization.py       # draw_*, plot_* (thread-safe via Agg)
│   ├── sensitivity.py         # sensibilité x0 / mesh / paramètres
│   ├── utils.py               # historique CSV, meilleur design
│   └── meta_optimization.py   # grid search hyperparamètres DE
├── config/
│   ├── opt_config.json        # bornes + 3 modes (rapide/normal/profond)
│   └── last_inputs.json       # persistance des inputs GUI entre sessions
├── cache/                     # maillages .msh + T_current.dat
├── results/                   # historique, figures, designs
├── app.py                     # interface graphique Tkinter
├── main.py                    # pipeline CLI complet
├── user_interface.py          # calcul ponctuel CLI
├── run_sensitivity_study.py   # études de sensibilité
├── comparison_methods.ipynb   # étude comparative (7 études)
├── requirements.txt           # dépendances Python
└── README.md
```

### 5.2. Interface graphique (`app.py`)

L'application Tkinter utilise un `ttk.Notebook` organisant le travail en trois onglets :

1. **Calcul ponctuel** : saisie des six paramètres physiques avec affichage des bornes ; choix de la taille du maillage ou import d'un `.msh` externe ; boutons *Calculer J* (sortie texte seule) et *Calculer J + afficher la solution* (qui bascule automatiquement sur l'onglet Visualisation).
2. **Optimisation** : trois modes pré-configurés (*Rapide* : mesh = 25, maxiter = 5, popsize = 4 ; *Normale* : mesh = 50, maxiter = 15, popsize = 8 ; *Approfondie* : mesh = 80, maxiter = 30, popsize = 15). Une barre de progression déterministe est mise à jour par un callback transmis à l'optimiseur.
3. **Visualisation** : six modes de tracé via boutons radio — maillage natif avec bords colorés par label ; champ $T$ courant ; champs $T$ initial et optimisé séparés ; comparaison côte à côte ; courbe de convergence. matplotlib est embarqué dans Tk via `FigureCanvasTkAgg` avec barre d'outils.

Quatre fonctionnalités améliorent l'expérience utilisateur au-delà du strict cahier des charges :

- **Persistance inter-sessions** : les valeurs saisies (paramètres, mesh, mode) sont sauvegardées dans `config/last_inputs.json` à chaque calcul et à la fermeture de la fenêtre, puis restaurées au lancement suivant.
- Bouton **« Charger le meilleur design »** qui pré-remplit les six entrées avec le meilleur $x$ trouvé dans l'historique CSV.
- **Auto-détection du maillage** utilisé pour produire un fichier $T$ en comparant les coordonnées $(x, y)$ du fichier avec les sommets des `.msh` en cache. Ceci évite les rendus erronés (« hexagone convexe ») lorsque l'utilisateur change de taille de maillage entre exécutions.
- **Threads de calcul découplés du thread Tk** : les sauvegardes PNG passent par `FigureCanvasAgg` directement (pas de `pyplot`), ce qui supprime des crashs liés à la non-réentrance de TkAgg sur macOS.

### 5.3. Pipeline en ligne de commande (`main.py`)

`main.py` exécute le pipeline complet en mode batch : génération du maillage ; évaluation du design initial ($x_0 = 0{,}5$) avec export du champ $T$ associé ; exécution successive de DE, NM et BH avec les budgets définis dans `config/opt_config.json` ; production des figures de convergence par méthode et d'une figure de convergence comparée ; export du champ $T$ optimisé et tracé de la comparaison initial vs optimisé. Toutes les sorties (CSV et PNG) sont déposées dans `results/`.

---

## 6. Cadre d'analyse numérique : `comparison_methods.ipynb`

Le rapport final reposera sur les résultats produits par ce notebook, qui est organisé en sept études autonomes couvrant l'ensemble des questions soulevées par la grille d'évaluation (Task 5 du cahier des charges) ainsi que les extensions encouragées (Task 10). Toutes les sorties (CSV et PNG) sont déposées dans `results_compare/`.

| # | Étude | Question principale | Mesh | Coût |
|---|---|---|---|---|
| 1 | Comparaison 3 méthodes (budget équivalent) | Quelle méthode est la plus efficiente ? | 50 | 3-5 min |
| 2 | Impact de la résolution du maillage | $J^\star$ converge-t-il à mesh fin ? | 15-60 | 5-10 min |
| 3 | Sensibilité au point initial (NM, BH) | Méthode locale fiable sans bon $x_0$ ? | 25 | 1-2 min |
| 4 | Sensibilité à la graine (DE, BH) | Variabilité aléatoire significative ? | 25 | 1-2 min |
| 5 | Hyperparamètres DE (popsize, mutation, strategy) | Quel réglage donne le meilleur $J^\star$ ? | 25 | 5-7 min |
| 6 | Hyperparamètres BH (stepsize, $T$ Metropolis) | Effets concrets de ces leviers ? | 25 | 3-4 min |
| 7 | Designs optimaux et champs $T$ | Convergence vers le même optimum physique ? | 50 | <1 min |

**Particularité méthodologique des études 5 et 6** : la sortie d'un algorithme stochastique pour une seule graine est par essence aléatoire. Conclure « `popsize` = 6 est meilleur que `popsize` = 10 » à partir d'un unique run pourrait n'être qu'un artefact du tirage initial. Chaque configuration d'hyperparamètre est donc évaluée sur `N_SEEDS_HP = 3` graines indépendantes ; la moyenne et l'écart-type de $J^\star$ sont reportés sous forme de barres d'erreur. Ce protocole permet de juger si l'effet observé d'un hyperparamètre est statistiquement significatif ou s'il est noyé dans la variabilité stochastique.

Les calculs ne sont pas encore tous exécutés à la date de ce rapport intermédiaire. La version finale du document intégrera :

- le tableau de synthèse de l'Étude 1 ($J^\star$, gain vs $J_0$, `n_eval`, temps, temps/évaluation) et la figure de Pareto associée ;
- les courbes $J^\star = f(\text{mesh})$ et $\text{temps} = f(\text{mesh})$ de l'Étude 2 ;
- les boxplots de variabilité (Études 3 et 4) avec discussion sur le caractère uni-modal ou multi-modal de la fonctionnelle $J$ ;
- les courbes avec barres d'erreur des Études 5 et 6 et la sélection des hyperparamètres optimaux pour DE et BH ;
- la comparaison qualitative et quantitative des designs trouvés par les trois méthodes (Étude 7), accompagnée des cartes de température correspondantes (échelle de couleurs partagée).

---

## 7. Résultats préliminaires

À ce stade, les éléments suivants ont déjà été validés :

- Le solveur FreeFEM++ s'exécute sans erreur pour $x = (0{,}5\ ;\ 0{,}5\ ;\ 0{,}5\ ;\ 0{,}5\ ;\ 0{,}5\ ;\ 0{,}5)$ à mesh = 50 et produit un champ $T$ continu ($T = 1$ sur la base, décroissance exponentielle dans le tronc et les ailettes).
- Le maillage natif `buildmesh` comporte **1 151 sommets et 1 760 triangles à mesh = 50**, contre 419 sommets à mesh = 25 et 2 200+ sommets à mesh = 80.
- L'optimisation rapide (DE, `maxiter = 5`, `popsize = 4`, mesh = 25) converge en moins d'une minute vers un design où $T$ moyen passe d'environ 0,09 (design uniforme) à 0,63 (design optimisé), soit un facteur 7×.
- L'interface graphique a été testée sur macOS avec le venv Homebrew (Python 3.14 + `python-tk@3.14`) et fonctionne pour les trois onglets.

La version finale du rapport remplacera cette section par les tableaux et figures issus de l'exécution complète du notebook, avec les valeurs exactes et les analyses statistiques associées.

---

## 8. Conclusion et perspectives

L'état actuel du projet couvre l'intégralité des tâches obligatoires du cahier des charges :

- **Task 1** (solveur FreeFEM++) : maillage de qualité avec `buildmesh`, conditions aux limites correctement labellisées, formulation variationnelle P1 standard.
- **Task 2** (module d'évaluation) : $J$ calculé directement dans `solver.edp` et exposé via stdout.
- **Task 3** (module d'optimisation) : trois algorithmes (DE, NM, BH) avec une interface uniforme et historique complet.
- **Task 4** (intégration plateforme) : interface graphique avec trois modes d'optimisation, visualisation embarquée, persistance des saisies.
- **Task 5** (analyse numérique) : cadre d'étude en sept volets prêt à être exécuté.

Les améliorations non encore implémentées et envisagées pour la suite sont :

- exécution complète du notebook et intégration des résultats dans le rapport final ;
- étude *material-selection discret* mentionnée au Task 10 du cahier des charges : remplacement des $k_i$ continus par un choix discret dans un catalogue de matériaux (typiquement $\{0{,}1\ ;\ 0{,}3\ ;\ 0{,}5\ ;\ 0{,}7\ ;\ 1{,}0\}$) ;
- construction d'un modèle de substitution (surrogate) de type krigeage entraîné sur l'historique des évaluations DE pour accélérer la convergence ;
- préparation de la présentation orale.

Le code source est versionné sous git et conserve la traçabilité de toutes les évolutions décrites dans ce rapport.

---

## Annexe — Références et environnement

### Références

[1] Hassan Ballout, Yvon Maday, Christophe Prud'homme. *Nonlinear compressive reduced basis approximation for multi-parameter elliptic problem*. In *Multiscale, Nonlinear and Adaptive Approximation II*, pages 55–73. Springer, 2024.

### Environnement logiciel

- Python ≥ 3.10 (testé sur 3.14 / macOS avec Homebrew).
- FreeFEM++ ≥ 4.13 (testé avec 4.15).
- Dépendances Python : `numpy ≥ 1.22`, `scipy ≥ 1.10`, `pandas ≥ 1.5`, `matplotlib ≥ 3.5`, `tqdm ≥ 4.60`, `python-dotenv ≥ 1.0`.
- Tkinter : livré avec Python sur Windows et macOS ; `sudo apt install python3-tk` sur Linux Debian/Ubuntu.
