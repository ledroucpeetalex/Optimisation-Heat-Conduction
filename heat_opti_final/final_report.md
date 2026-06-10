# Development of an Automatic Design Optimization Platform for a 2D Heat-Conduction System Using FreeFEM++

**HEAT-COND benchmark — FreeFEM++ and Python**

*Final Report*

| Course | Optimization and Numerical Analysis (MATH6304P-260-M01) — Spring 2026 |
|---|---|
| Institution | Shanghai Jiao Tong University — SPEIT |
| Instructor | Prof. Helin Gong |
| Teaching Assistant | Zhipu Cui |
| Authors | Laurent Zhu, Alexandre Le Droucpeet |

---

## 1. Introduction and Project Objective

This project develops a compact, automatic design-optimization platform for a two-dimensional steady-state heat-conduction problem, the **HEAT-COND benchmark** drawn from the reduced-basis literature [1]. The platform integrates, within a single reproducible workflow, the five components required by the assignment: a finite-element solver written in FreeFEM++, an objective-evaluation module, a Python optimization driver, a graphical user interface, and a numerical-analysis framework for comparing optimization strategies.

The scientific goal is to couple the solution of an elliptic partial differential equation (PDE) posed on a non-trivial 2D geometry with an automated parametric optimization loop, and to determine, through systematic experiments, **which optimization strategy is best suited to this problem and why**. As we show, the objective functional is smooth and unimodal with its optimum located on the boundary of the admissible set; this single structural fact explains the entire ranking of the methods we tested, and motivates the optimizer chosen as the platform default.

The report follows the structure required by the assignment: the mathematical model (Section 2), the numerical formulation and FreeFEM++ implementation (Section 3), the optimization strategy (Section 4), numerical results (Section 5), analysis and discussion (Section 6), and conclusions with possible improvements (Section 7).

---

## 2. Mathematical Model

### 2.1 Geometry

The computational domain $\Omega \subset [0,1]^2$ is a comb-shaped region consisting of a central vertical **spine** of width $w_s = 0.10$ and five pairs of horizontal **fins** of thickness $t_f = 0.06$, centred at the vertical positions $y \in \{0.16, 0.32, 0.48, 0.64, 0.80\}$. The inner edges of the spine are at $x_g = (1 - w_s)/2 = 0.45$ and $x_d = x_g + w_s = 0.55$. The geometry is symmetric about $x = 0.5$ and multiply-connected through the gaps between fins. Two boundary regions are distinguished: $\Gamma_{\text{Base}}$, the bottom edge of the spine $\{(x,0): x\in[x_g,x_d]\}$, and $\Gamma_{\text{Fin}}$, the remainder of $\partial\Omega$ (the spine top, the outer fin tips, and all fin flanks).

### 2.2 Governing Equation

The dimensionless temperature $T$ satisfies the steady-state heat-conduction equation with mixed (Dirichlet–Robin) boundary conditions:

$$
\begin{cases}
-\nabla \cdot \big(k(\mathbf{x})\,\nabla T\big) = 0 & \text{in } \Omega, \\[2pt]
T = 1 & \text{on } \Gamma_{\text{Base}}, \\[2pt]
k(\mathbf{x})\,\dfrac{\partial T}{\partial n} + \mathrm{Bi}\,T = 0 & \text{on } \Gamma_{\text{Fin}}.
\end{cases}
$$

The conductivity $k(\mathbf{x})$ is piecewise constant: $k = 1$ in the spine, and $k = k_i$ ($i=1,\dots,5$) in fin $i$, counted bottom to top. The Biot number $\mathrm{Bi}$ controls the convective coupling on $\Gamma_{\text{Fin}}$.

### 2.3 Objective and Optimization Problem

The objective is to maximize the average temperature on the fin boundary,

$$
J(\mathbf{x}) = \frac{1}{|\Gamma_{\text{Fin}}|} \int_{\Gamma_{\text{Fin}}} T \, \mathrm{d}\Gamma,
$$

over the design vector $\mathbf{x} = (k_1, k_2, k_3, k_4, k_5, \mathrm{Bi}) \in \mathbb{R}^6$, subject to $k_i \in [0.1, 1.0]$ and $\mathrm{Bi} \in [0.01, 1.0]$:

$$
\max_{\mathbf{x}} \; J(\mathbf{x}) \quad \text{subject to box constraints.}
$$

Physically, a high $J$ means the spine transfers the heat injected at the base ($T = 1$) efficiently to the peripheral fin surfaces.

---

## 3. Numerical Formulation and FreeFEM++ Implementation

### 3.1 Mesh Generation

The mesh is constructed explicitly with FreeFEM's `buildmesh` rather than by truncating a background grid, so that element edges align exactly with the geometry. The boundary is described as a closed, counter-clockwise polyline of **44 segments / 45 points**, traversed by a single indexed `border` definition; the number of subdivisions of each segment is proportional to its Euclidean length times a global density parameter `meshsize`. This produces straight, geometry-aligned boundaries and a clean Delaunay triangulation inside, avoiding the staircase edges and mislabelled facets of a naive `square(...)` + `trunc(...)` approach. Two boundary labels are assigned: **label 1** for the base segment (Dirichlet) and **label 2** for the rest of the boundary (Robin). Each mesh is written once to `cache/mesh_<size>.msh` and reused. As a reference, the mesh at density 50 contains **1 151 vertices and 1 760 triangles** (419 vertices at density 25).

### 3.2 PDE Solver

`solver.edp` reads its parameters through `getARGV` (`-meshfile`, `-k1`…`-k5`, `-Bi`, optional `-Tout`). The conductivity field is built as a piecewise-constant indicator over the regions, and the weak form is assembled in the $P_1$ finite-element space $V_h$: find $T \in V_h$ with $T = 1$ on $\Gamma_{\text{Base}}$ such that, for all test functions $v$,

$$
a(T, v) = \int_\Omega k \, \nabla T \cdot \nabla v \, \mathrm{d}\Omega + \int_{\Gamma_{\text{Fin}}} \mathrm{Bi}\, T\, v \, \mathrm{d}\Gamma = 0.
$$

The Dirichlet condition is imposed strongly via `on(1, T = 1)`; the Robin condition enters naturally as the boundary integral over label 2. The objective $J = \int_{\Gamma_2} T\,\mathrm{d}\Gamma \,/\, \int_{\Gamma_2} 1\,\mathrm{d}\Gamma$ is computed directly inside the solver and printed to standard output as `Objective_J=<value>`, removing any intermediate text files.

### 3.3 Python Coupling and Platform Integration

The Python layer (`src/freefem_interface.py`) drives FreeFEM through `subprocess`: `ensure_mesh` generates and caches meshes, and `run_solver(x, mesh_size, t_out=None)` invokes `solver.edp`, parses `Objective_J=` from stdout, and optionally exports the temperature field. The full platform comprises an input/configuration module (`config/opt_config.json`), the FreeFEM simulation module, the objective-evaluation module, the optimization driver (`src/optimization.py`), a result-saving and visualization module, a Tkinter graphical interface (`app.py`) with an algorithm selector, and a command-line pipeline (`main.py`).

### 3.4 Solver Verification

To verify the implementation independently, the $P_1$ weak form was re-implemented in pure NumPy (`standalone_compare/fem_solver.py`) and compared against FreeFEM on identical cached meshes. At density 50 the two solvers agree to six significant digits at the optimal corner — $J(1,1,1,1,1,0.01) = 0.730396$ in both — and the vertex/triangle counts match exactly. This cross-check validates the assembled operator and the boundary-integral objective, and it lets the optimizer comparison of Section 5 be reproduced without a FreeFEM installation.

---

## 4. Optimization Strategy

The optimization driver wraps every algorithm behind a uniform interface that returns a normalized dictionary (`best_J`, `best_x`, `n_eval`, `time`, `success`, full evaluation `history`) and records every PDE evaluation. Random-based methods use a fixed seed (`seed = 42`) for reproducibility. After a preliminary screening we retained **three complementary methods**, one per algorithmic family, for all comparisons:

- **Nelder-Mead (NM)** — a local, deterministic simplex method in dimension 6; gradient-free, sensitive to the starting point, very cheap.
- **Differential Evolution (DE)** — a global, population-based, gradient-free method (`popsize × 6` individuals, mutation and crossover); robust to multimodality but the most expensive.
- **Adam → L-BFGS-B** — a two-phase gradient hybrid: Adam (finite-difference gradient, per-coordinate adaptive step) drives the iterate into a good region, then L-BFGS-B refines with native bound handling.

Because the objective is evaluated by a black-box PDE solver, the gradient-based phase estimates the gradient by central finite differences, which costs $2 \times 6 = 12$ extra evaluations per gradient in this 6-dimensional problem — a cost we return to in Section 6.

The experimental design is implemented in the notebook `comparison_methods.ipynb` as **seven self-contained studies** covering the analysis points of the assignment (Task 5) and the encouraged extensions (Task 10). To prevent stochastic artifacts, every hyperparameter configuration in the DE study is averaged over three independent seeds, and mean ± standard deviation are reported.

---

## 5. Numerical Results

All figures are produced by the notebook into `results_compare/`. The quantitative values below come from the verified $P_1$ solver of Section 3.4 and from the FreeFEM runs of the notebook. The reference initial design is the uniform vector $\mathbf{x}_0 = (0.5,\dots,0.5)$, for which $J_0 \approx 0.077$.

### 5.1 Comparison of the Three Methods and the Optimal Design (Study 1)

Starting from the common point $\mathbf{x}_0 = 0.5$ (density 25; the ranking is identical at density 50):

| Method | $J^\star$ | $n_{\text{eval}}$ | Time (s) | Design $\mathbf{x}^\star$ |
|---|---|---|---|---|
| **Nelder-Mead** | **0.734167** | **115** | 0.28 | $(1,1,1,1,1,0.01)$ |
| Adam → L-BFGS | 0.734167 | 373 | 0.90 | $(1,1,1,1,1,0.01)$ |
| Differential Evolution | 0.734110 | 768 | 1.86 | $(1,0.996,0.986,1,1,0.01)$ |

All three methods reach the same global optimum,

$$
\mathbf{x}^\star = (k_1,\dots,k_5,\mathrm{Bi}) = (1,\,1,\,1,\,1,\,1,\,0.01),
$$

i.e. **all conductivities at their upper bound and the Biot number at its lower bound** — a vertex of the admissible box. At the reference density 50 this corner gives $J^\star = 0.730396$ (verified solver), an improvement of roughly $850\%$ over $J_0$. Nelder-Mead reaches the optimum at the lowest cost (115 evaluations); Differential Evolution is the most expensive and stops just short of the exact corner. (Figures `etude1_convergence.png`, `etude1_pareto.png`.)

### 5.2 Mesh-Sensitivity (Study 2, Task 10)

The optimized objective (Nelder-Mead) as a function of mesh density:

| Mesh density | 15 | 25 | 40 | 60 |
|---|---|---|---|---|
| $J^\star$ | 0.7414 | 0.7342 | 0.7313 | 0.7295 |

$J^\star$ decreases monotonically and stabilizes near $0.73$ as the mesh is refined, indicating approximate mesh-independence by density 50–60; coarse meshes slightly over-estimate the objective. The wall-clock time grows roughly quadratically with the density, consistent with the increase in degrees of freedom. (Figure `etude2_mesh.png`.)

### 5.3 Robustness (Studies 3 and 4)

Study 3 varies the starting point for the local and gradient methods (NM, Adam→L-BFGS); Study 4 varies the random seed for the only stochastic method (DE). Nelder-Mead is extremely reproducible across starting points — its standard deviation in $J^\star$ is below $10^{-3}$ (mean $0.734$) — confirming that the functional is effectively unimodal: any reasonable start reaches the same optimum. Adam→L-BFGS likewise reaches the corner from every tested start. Differential Evolution shows a small seed-to-seed spread (standard deviation $\approx 2\times10^{-3}$); all three methods are robust here. (Figure `etude34_robustness.png`.)

### 5.4 Differential-Evolution Hyperparameters (Study 5)

Averaged over three seeds: a larger `popsize` improves both the mean objective and its stability (from $0.672 \pm 0.063$ at `popsize = 2` to $0.717 \pm 0.002$ at `popsize = 15`); a **low mutation** factor is clearly preferable ($0.718$ at $F = 0.3$ versus $0.611$ at $F = 1.8$); and among strategies `best1bin` performs best ($0.701$). These trends are consistent with an exploitation-dominated, unimodal landscape. (Figure `etude5_DE_hyperparams.png`.)

### 5.5 The Adam → L-BFGS Hybrid and the Value of Preconditioning (Study 7)

This study isolates the contribution of each phase of the hybrid by adding two diagnostic runs from the common start $\mathbf{x}_0 = 0.5$:

| Variant | $J^\star$ | $n_{\text{eval}}$ | Reaches the corner |
|---|---|---|---|
| Adam → L-BFGS | 0.734167 | 373 | yes |
| *Adam alone* | 0.734167 | 360 | yes |
| *L-BFGS-B alone* | 0.721633 | 106 | **no — stalls** |

**L-BFGS-B alone stalls** at $J = 0.722$ with the conductivities stuck near $0.5$, whereas the hybrid reaches the corner. A direct probe of the gradient at the stall point explains why: there

$$
|\partial J / \partial \mathrm{Bi}| \approx 19.4 \qquad\text{but}\qquad |\partial J / \partial k_i| \approx 0.009,
$$

a ratio of about $2000{:}1$. Once $\mathrm{Bi}$ saturates at its lower bound, the objective is almost flat in the conductivity directions, and the pure quasi-Newton method declares convergence. Adam, which normalizes each coordinate by the running root-mean-square of its gradient, turns this tiny but consistent signal into full-sized steps and drives the conductivities to $1.0$. This is a textbook illustration of adaptive preconditioning on an ill-conditioned problem, and it is exactly why the first phase of the hybrid is necessary. (Figure `etude7_convergence.png`; data `etude7_adam_lbfgs.csv`.)

Study 6 (not tabulated here) confirms that all three methods converge to the same physical design and temperature field; the corresponding maps are saved as `etude6_T_fields.png`.

---

### 5.6 Extension: Material-Cost Trade-off

The optimum above is a vertex of the box because higher conductivity is free. A more realistic design problem penalizes material: we maximize the scalarized objective $F_\lambda(\mathbf{x}) = J(\mathbf{x}) - \lambda\bar{k}$, where $\bar{k} = \frac15\sum_i k_i$ is the mean conductivity (a proxy for material cost), and sweep $\lambda \ge 0$ to trace the performance--cost Pareto front. The penalty is evaluated in Python around the solver, so it ports unchanged to FreeFEM and is exposed in the graphical interface through a single $\lambda$ field.

Two findings stand out. First, the front is strongly concave near the optimum: one can remove $32\%$ of the conductive material for only a $0.8\%$ loss of performance (the knee at $\bar{k} = 0.68$, $J = 0.728$ versus $J_{\max} = 0.734$). Second, the optimal design becomes non-uniform with a clear physical ordering $k_1 > k_2 > k_3 > k_4 > k_5$: the fins nearest the hot base retain their conductivity as $\lambda$ grows, while the upper fins are economized first; the Biot number stays at its lower bound throughout. This turns the otherwise trivial corner problem into a genuine multi-objective design study. (Figures `ext_material_pareto_front.png`, `ext_material_design_vs_lambda.png`.)

### 5.7 Optimizer Comparison on the Penalized Problem

Section 5.6 traced the front with a single optimizer; we repeat the $\lambda$ sweep with all three methods. The three Pareto fronts essentially coincide: Nelder-Mead and Adam$\to$L-BFGS overlap, while Differential Evolution lies marginally inside the front in the mid-cost region (budget-limited search). At a representative weight $\lambda = 0.05$ (near the knee), Nelder-Mead and Adam reach the same scalarized optimum $F^\star = 0.696$, with Nelder-Mead the cheapest (178 vs 304 evaluations); DE attains $F^\star = 0.695$ at 324 evaluations. The pure-performance ranking therefore carries over to the constrained problem: the methods agree on the trade-off, and Nelder-Mead remains the most efficient default. (Figures `ext_methods_pareto.png`, `ext_methods_convergence.png`.)

| Method | $J$ | cost $\bar k$ | $F = J-\lambda\bar k$ | $n_{\text{eval}}$ |
|---|---|---|---|---|
| **Nelder-Mead** | 0.7206 | 0.490 | **0.6961** | **178** |
| Adam $\to$ L-BFGS | 0.7212 | 0.502 | 0.6961 | 304 |
| Differential Evolution | 0.7237 | 0.581 | 0.6947 | 324 |

## 6. Analysis and Discussion

**Physical interpretation of the optimum.** The optimal design $(k = 1,\ \mathrm{Bi} = 0.01)$ is physically intuitive: maximizing every conductivity lets the spine conduct the base heat to the fins with minimal internal resistance, while minimizing the Biot number minimizes the convective loss at the fin surfaces, so the fins remain hot. The objective is monotone increasing in each $k_i$ and monotone decreasing in $\mathrm{Bi}$; consequently the maximizer sits at a corner of the admissible hypercube, and the functional is smooth and unimodal. This single structural property explains every result above.

**Why Nelder-Mead wins here.** For a smooth, unimodal objective with a boundary optimum, a local simplex started from the centre slides directly to the corner. It is the most accurate, the cheapest (115 evaluations), and by far the most reproducible (standard deviation $< 10^{-3}$). It is therefore the platform's recommended default.

**Why all three are kept.** Nelder-Mead is a *local* method and cannot, by itself, certify global optimality. Differential Evolution, a global population method, lands independently at the same corner; it is this agreement that justifies *trusting* the Nelder-Mead result as the global optimum. The Adam → L-BFGS hybrid contributes the complementary lesson that adaptive preconditioning is what makes a gradient method succeed on this ill-conditioned problem, where a pure quasi-Newton method stalls. We emphasize that the Nelder-Mead advantage is **problem-specific**: on a multimodal objective, the global method would be preferred; on an ill-conditioned problem without a boundary optimum, the gradient hybrid would be.

**Computational cost and bottlenecks.** The dominant cost is the FreeFEM PDE solve invoked through a subprocess (about $0.23$ s per evaluation at density 50). For the gradient-based phase, the finite-difference gradient adds a factor of twelve per gradient in 6D, which is why Adam → L-BFGS, although effective, is more expensive than Nelder-Mead. The most promising accelerations are therefore (i) an analytical or adjoint gradient and (ii) a surrogate model trained on the evaluation history.

**Advantages and limitations of the platform.** The platform is complete and reproducible: a cached meshing stage, a clean $P_1$ solver verified against an independent implementation, a uniform optimizer interface with full history logging, a graphical interface with an algorithm selector (NM / DE / Adam → L-BFGS), and a seven-study analysis notebook. Its main limitations are the subprocess-based coupling, which bounds throughput, and the absence of an analytical gradient; both are natural directions for future work.

---

## 7. Conclusion and Possible Improvements

We have built an automatic design-optimization platform for the HEAT-COND benchmark that satisfies all required tasks: a geometry-aligned FreeFEM++ $P_1$ solver (Task 1), an objective module computed directly in the solver and exposed through stdout (Task 2), an optimization driver coupling the three retained methods behind a uniform interface (Task 3), full platform integration with a graphical interface and a command-line pipeline (Task 4), and a structured numerical analysis across seven studies (Task 5). The encouraged extensions — comparison of several optimizers, mesh-sensitivity, parameter-sensitivity, and a gradient hybrid — are all included.

The central finding is that the HEAT-COND objective is smooth and unimodal with a boundary optimum at $\mathbf{x}^\star = (1,1,1,1,1,0.01)$, giving $J^\star \approx 0.730$ at density 50, an improvement of roughly $850\%$ over the uniform design. On this landscape **Nelder-Mead is the most efficient optimizer and the recommended default**, with Differential Evolution retained as a global-optimality check and the Adam → L-BFGS hybrid demonstrating the value of adaptive preconditioning where a pure quasi-Newton method stalls.

Possible improvements include: (i) a **discrete material-selection** formulation replacing the continuous $k_i$ by a catalogue such as $\{0.1, 0.3, 0.5, 0.7, 1.0\}$; (ii) a **surrogate / response-surface model** (e.g. Kriging) trained on the evaluation history to accelerate convergence; and (iii) an **adjoint gradient** for the conductivities and the Biot number, which would remove the finite-difference overhead and turn the gradient hybrid into the cheapest option.

---

## Member Contributions

*(To be confirmed by the team before submission.)* Both members contributed substantially to the overall methodology. Laurent Zhu focused on the FreeFEM++ solver and meshing, the Python–FreeFEM coupling, and the optimization driver and its experimental studies. Alexandre Le Droucpeet focused on the mathematical model, the graphical interface and visualization, the analysis of the numerical results, and the report.

---

## Appendix — References and Software Environment

**Reference**

[1] H. Ballout, Y. Maday, C. Prud'homme. *Nonlinear compressive reduced basis approximation for multi-parameter elliptic problem*. In *Multiscale, Nonlinear and Adaptive Approximation II*, pp. 55–73. Springer, 2024.

**Software environment**

- Python ≥ 3.10 (tested on 3.14, macOS, Homebrew).
- FreeFEM++ ≥ 4.13 (tested with 4.15).
- Python dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `tqdm`, `python-dotenv`; Tkinter ships with Python on macOS/Windows.

**Reproducibility**

The full analysis is reproduced by `comparison_methods.ipynb` (requires FreeFEM++). The optimizer comparison of Section 5 can be reproduced without FreeFEM via `standalone_compare/compare_opt.py`, which uses the NumPy solver verified against FreeFEM in Section 3.4.
