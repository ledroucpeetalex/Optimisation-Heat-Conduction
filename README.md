# Optimisation: Heat Conduction (HEAT-COND)

Automatic design optimization platform for a 2D steady heat conduction problem (the
**HEAT-COND** benchmark). It couples a **FreeFEM++** finite element solver ($P_1$) with a
**Python** optimization driver.

> Course project, _Optimization and Numerical Analysis_ (MATH6304P-260-M01), SJTU SPEIT, Spring 2026.
> **Authors: Laurent ZHU, Alexandre LE DROUCPEET.** Instructor: Prof. Helin Gong. TA: Zhipu Cui.

## Two-part approach

The report (`report/main.pdf`) and the code follow the same progression.

**Part I, initial problem (simple objective J).** We solve
$-\nabla\cdot(k\,\nabla T)=0$ on a comb shaped domain (a spine plus five pairs of fins), with
$T=1$ on the base (Dirichlet) and a Robin condition (Biot number) on the rest of the boundary.
We **maximize the mean temperature J on the fin boundary** over $(k_1,\dots,k_5,\mathrm{Bi})$.
The objective is smooth and unimodal, with a boundary optimum at $x^\star=(1,1,1,1,1,0.01)$ and
$J^\star\approx0.730$. On this landscape Nelder-Mead is the most efficient method.

**Part II, parametric problem (complex objective).** Each fin now carries a **discrete material**
(conductivity, **price**, density) and a **continuous geometry** (thickness, length). We maximize a
multi-criteria objective
$F = Q - \lambda_{\text{cost}}\,\text{Cost} - \lambda_{\text{mass}}\,\text{Mass}$
(16 variables, mixed integer and continuous). We apply the **same methods as in Part I**
(Nelder-Mead, Differential Evolution, Adam then L-BFGS) plus **Bayesian optimization**. They land
within a few percent of one another; **Bayesian optimization gives the best objective with by far the
fewest evaluations**, so it is retained as the default method. We then characterize the objective
evolution (per evaluation and per CPU second) and the performance/cost Pareto front.

## Repository layout

```
.
├── freefem/                 # FreeFEM++ solvers (.edp)
│   ├── mesh.edp             # comb mesh, fixed geometry          (Part I)
│   ├── solver.edp           # P1 solver, objective J             (Part I)
│   ├── mesh_param.edp       # mesh parameterized by (t_i, l_i)   (Part II)
│   └── solver_param.edp     # P1 solver, dissipated heat Q (+ J) (Part II)
├── heatcond/                # Python package (core platform)
│   ├── config.py            # paths and lazy FreeFEM access
│   ├── freefem.py           # subprocess interface: simple and parametric solvers
│   ├── reference_solver.py        # pure NumPy P1 solver (simple model, validation)
│   ├── reference_solver_param.py  # pure NumPy P1 solver (parametric model, no FreeFEM)
│   ├── materials.py         # material catalogue, design vector, cost and mass
│   ├── objective.py         # SimpleObjective (J) and ParametricObjective (F), with history
│   ├── optimizers/
│   │   ├── algorithms.py    # generic optimizer toolbox: NM, DE, L-BFGS-B (scipy), Adam, Bayesian (skopt)
│   │   ├── simple.py        # method drivers wired to the FreeFEM simple model (Part I)
│   │   └── parametric.py    # method drivers + multi-fidelity, parametric model (Part II)
│   ├── sensitivity.py       # sensitivity studies (start point, mesh, parameter sweep)
│   ├── meta_optimization.py # grid search of DE hyperparameters
│   ├── visualization.py     # plots: mesh, T field, convergence
│   └── utils.py
├── apps/                    # graphical interfaces (Tkinter)
│   ├── gui_simple.py        # Part I
│   └── gui_param.py         # Part II (method selector: NM, DE, Adam then L-BFGS, multi-fidelity)
├── scripts/                 # command line entry points
│   ├── run_simple.py        # Part I pipeline (three methods + figures)
│   ├── run_param.py         # Part II pipeline (DE then refinement)
│   ├── sensitivity_study.py # sensitivity studies
│   ├── build_part1_notebook.py  # (re)generate the Part I notebook
│   ├── build_part2_notebook.py  # (re)generate the Part II notebook
│   ├── reproduce_part1_no_freefem.py  # Part I optimizer comparison without FreeFEM
│   ├── plot_objective_evolution.py    # objective evolution figure (Part II)
│   └── cli_eval_simple.py   # evaluate J for one (k1..k5, Bi)
├── notebooks/               # reproducible studies (scipy + NumPy reference solver, no FreeFEM)
│   ├── part1_simple_objective.ipynb
│   └── part2_material_geometry.ipynb
├── results/                 # generated figures and data
│   ├── part1/               # Part I figures (part1_*, geometry_mesh) and summary.json
│   └── part2/               # Part II figures (part2_*) and methods_param.json
├── config/opt_config.json   # bounds and optimizer settings (Part I)
└── report/                  # final report (LaTeX and PDF)
```

**Deliverable: [`report/main.pdf`](report/main.pdf)** (source `report/main.tex`, compiles on Overleaf
with the `figures/` folder).

## Running the application

The graphical apps use the real FreeFEM++ solver, so FreeFEM must be installed.

```bash
# 1. Python dependencies (Python >= 3.10; tkinter ships with Python)
pip install -r requirements.txt

# 2. FreeFEM++ (>= 4.13): install it from https://freefem.org, then tell the project where it is.
#    Create a file named .env in the project root with a single line giving the absolute
#    path to the FreeFem++ binary. On macOS, for example:
echo 'FREEFEM_PATH="/Applications/FreeFem++.app/Contents/ff-4.15/bin/FreeFem++"' > .env

# 3. Launch an app
python apps/gui_param.py     # Part II: material + geometry (method selector + multi-fidelity)
python apps/gui_simple.py    # Part I: simple model (k1..k5, Bi)
```

Command-line equivalents (also require FreeFEM and the `.env`):

```bash
python scripts/run_param.py        # Part II pipeline (writes results/part2/best_param.json)
python scripts/run_simple.py       # Part I pipeline (writes results/part1/...)
python scripts/cli_eval_simple.py  # evaluate J for one (k1..k5, Bi) typed in
```

## Reproducing the report results

The studies and figures of both parts run on the validated pure-NumPy reference solver, so they need
**no FreeFEM**. Only NumPy, SciPy, pandas and matplotlib are required, plus `scikit-optimize` for the
Bayesian-optimization study.

```bash
# 1. Python dependencies, including scikit-optimize (already listed in requirements.txt)
pip install -r requirements.txt

# 2. Run both notebooks and copy every figure into report/figures/, in one step
python scripts/refresh_figures.py
```

What `scripts/refresh_figures.py` does, step by step:

1. executes the code cells of `notebooks/part1_simple_objective.ipynb` and
   `notebooks/part2_material_geometry.ipynb` directly (no Jupyter required);
2. the Part I notebook writes `results/part1/part1_*.png` and `results/part1/summary.json`;
3. the Part II notebook writes `results/part2/part2_*.png` and `results/part2/methods_param.json`;
4. it copies all those PNGs into `report/figures/`.

## Requirements

- Python 3.10 or later (numpy, scipy, pandas, matplotlib, tqdm, python-dotenv, nbformat; tkinter ships with Python)
- FreeFEM++ 4.13 or later (https://freefem.org) for the apps and pipelines. Not needed to run the notebooks or to recompile the report.
