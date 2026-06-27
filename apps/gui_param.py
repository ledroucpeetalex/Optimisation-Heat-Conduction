#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graphical interface for the enriched HEAT-COND model (geometry + material).

Tkinter application. Three tabs:
  1. Single evaluation : Q, J, cost, mass for one design (material + t + l per
     fin, Bi).
  2. Optimization      : 4 methods (NM / DE / Adam->L-BFGS / Bayesian) + multi-fidelity
     (coarse DE -> fine refinement), objective F = Q - lam_cost*cost - lam_mass*mass.
  3. Visualization     : mesh + temperature field T (embedded matplotlib).

Threading + a progress bar keep the UI responsive.
"""

import os
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo root for `import heatcond`

from heatcond import config                      # noqa: E402
from heatcond import materials as model          # noqa: E402
from heatcond.optimizers import parametric as optimize  # noqa: E402
from heatcond.objective import ParametricObjective  # noqa: E402
from heatcond.freefem import (ensure_mesh_param as ensure_mesh,  # noqa: E402
                              run_solver_param,
                              read_temperature_field)
from heatcond.visualization import read_freefem_mesh, draw_mesh, draw_temperature  # noqa: E402

# ---------- Palette ----------
C_BG = "#f4f6f8"
C_CARD = "#ffffff"
C_PRIMARY = "#2c3e50"
C_ACCENT = "#2980b9"
C_MUTED = "#7f8c8d"
C_ERR = "#c0392b"

MAT_NAMES = [m["name"] for m in model.MATERIALS]
NAME_TO_IDX = {m["name"]: i for i, m in enumerate(model.MATERIALS)}

CACHE = config.CACHE_DIR
RESULTS = config.RESULTS_PART2


# ===========================================================================
class EnrichedGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("HEAT-COND enriched - geometry + material")
        self.root.geometry("1080x800")
        self.root.minsize(980, 720)
        self.root.configure(bg=C_BG)

        self.last_T_file = None
        self.last_mesh_path = None
        self.last_opt_T = None
        self.last_opt_mesh = None
        self.last_best_x = None

        self._setup_style()
        self._build_header()
        self._build_notebook()
        self._build_statusbar()

    # ---------- Style ----------
    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        base_font = tkfont.nametofont("TkDefaultFont")
        base_font.configure(size=10)
        self.root.option_add("*Font", base_font)
        style.configure(".", background=C_BG)
        style.configure("TFrame", background=C_BG)
        style.configure("Card.TFrame", background=C_CARD)
        style.configure("TLabel", background=C_BG, foreground=C_PRIMARY)
        style.configure("Muted.TLabel", background=C_BG, foreground=C_MUTED,
                        font=("TkDefaultFont", 9))
        style.configure("Title.TLabel", background=C_PRIMARY, foreground="white",
                        font=("TkDefaultFont", 16, "bold"))
        style.configure("Subtitle.TLabel", background=C_PRIMARY,
                        foreground="#bdc3c7", font=("TkDefaultFont", 10))
        style.configure("Status.TLabel", background="#dfe4e8",
                        foreground=C_PRIMARY, padding=(8, 4))
        style.configure("Result.TLabel", background=C_BG, foreground=C_PRIMARY,
                        font=("Courier", 10))
        style.configure("TLabelframe", background=C_BG)
        style.configure("TLabelframe.Label", background=C_BG, foreground=C_PRIMARY,
                        font=("TkDefaultFont", 10, "bold"))
        style.configure("TNotebook", background=C_BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 8))
        style.map("TNotebook.Tab",
                  background=[("selected", C_CARD), ("!selected", "#dfe4e8")],
                  foreground=[("selected", C_PRIMARY), ("!selected", C_MUTED)])
        style.configure("Accent.TButton", background=C_ACCENT, foreground="white",
                        padding=(12, 6), font=("TkDefaultFont", 10, "bold"),
                        borderwidth=0)
        style.map("Accent.TButton",
                  background=[("active", C_PRIMARY), ("disabled", "#95a5a6")])
        style.configure("TButton", padding=(10, 5))
        style.configure("Horizontal.TProgressbar", troughcolor="#dfe4e8",
                        background=C_ACCENT, thickness=14)

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=C_PRIMARY, height=72)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        inner = tk.Frame(hdr, bg=C_PRIMARY)
        inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=8)
        ttk.Label(inner, text="HEAT-COND - Enriched model (geometry + material)",
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(inner, text="Objective Q = dissipated heat  |  F = Q - λ_cost*cost - λ_mass*mass",
                  style="Subtitle.TLabel").pack(anchor="w")

    def _build_notebook(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=12, pady=(10, 6))
        self.nb = nb
        self.tab_eval = ttk.Frame(nb, padding=12)
        self.tab_opt = ttk.Frame(nb, padding=12)
        self.tab_viz = ttk.Frame(nb, padding=12)
        nb.add(self.tab_eval, text="  Single evaluation  ")
        nb.add(self.tab_opt, text="  Optimization  ")
        nb.add(self.tab_viz, text="  Visualization  ")
        self._build_tab_eval()
        self._build_tab_opt()
        self._build_tab_viz()

    def _build_statusbar(self):
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel",
                  anchor="w").pack(fill=tk.X, side=tk.BOTTOM)

    def _set_status(self, msg):
        self.status_var.set(msg)

    # =======================================================================
    # Tab 1 - Single evaluation
    # =======================================================================
    def _build_tab_eval(self):
        tab = self.tab_eval
        tab.columnconfigure(0, weight=1)

        df = ttk.LabelFrame(tab, text="Fin design (material / thickness t / length l)",
                            padding=12)
        df.grid(row=0, column=0, sticky="new")
        ttk.Label(df, text="Fin", style="Muted.TLabel").grid(row=0, column=0, padx=6)
        ttk.Label(df, text="Material", style="Muted.TLabel").grid(row=0, column=1, padx=6)
        ttk.Label(df, text=f"t ∈ [{model.T_LO}, {model.T_HI}]",
                  style="Muted.TLabel").grid(row=0, column=2, padx=6)
        ttk.Label(df, text=f"l ∈ [{model.L_LO}, {model.L_HI}]",
                  style="Muted.TLabel").grid(row=0, column=3, padx=6)

        self.mat_cbs, self.t_entries, self.l_entries = [], [], []
        for i in range(model.N_FINS):
            ttk.Label(df, text=f"{i + 1}").grid(row=i + 1, column=0, padx=6, pady=3)
            cb = ttk.Combobox(df, values=MAT_NAMES, state="readonly", width=12)
            cb.set("Copper")
            cb.grid(row=i + 1, column=1, padx=6, pady=3)
            et = ttk.Entry(df, width=8); et.insert(0, str(model.T_DEFAULT))
            et.grid(row=i + 1, column=2, padx=6, pady=3)
            el = ttk.Entry(df, width=8); el.insert(0, str(model.L_DEFAULT))
            el.grid(row=i + 1, column=3, padx=6, pady=3)
            self.mat_cbs.append(cb); self.t_entries.append(et); self.l_entries.append(el)

        # Bi + mesh
        pf = ttk.LabelFrame(tab, text="Condition & mesh", padding=12)
        pf.grid(row=1, column=0, sticky="new", pady=(10, 0))
        ttk.Label(pf, text=f"Bi ∈ [{model.BI_LO}, {model.BI_HI}] :").grid(row=0, column=0, sticky="w")
        self.bi_var = tk.StringVar(value="0.3")
        ttk.Entry(pf, textvariable=self.bi_var, width=10).grid(row=0, column=1, padx=8, sticky="w")
        ttk.Label(pf, text="mesh_size :").grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.mesh_size_var = tk.StringVar(value="40")
        ttk.Entry(pf, textvariable=self.mesh_size_var, width=8).grid(row=0, column=3, padx=8, sticky="w")

        bf = ttk.Frame(tab)
        bf.grid(row=2, column=0, sticky="we", pady=(12, 6))
        self.btn_calc = ttk.Button(bf, text="Compute Q / J / cost / mass",
                                   style="Accent.TButton", command=self.calc_point)
        self.btn_calc.pack(side=tk.LEFT)
        ttk.Button(bf, text="Show T field",
                   command=lambda: self._select_viz("T_current")).pack(side=tk.LEFT, padx=8)
        ttk.Button(bf, text="↻ Default design",
                   command=self.reset_design).pack(side=tk.RIGHT)

        rf = ttk.LabelFrame(tab, text="Result", padding=12)
        rf.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        tab.rowconfigure(3, weight=1)
        self.eval_result_var = tk.StringVar(value="Waiting…")
        ttk.Label(rf, textvariable=self.eval_result_var, style="Result.TLabel",
                  wraplength=950, justify="left").pack(anchor="w")

    # =======================================================================
    # Tab 2 - Optimization
    # =======================================================================
    def _build_tab_opt(self):
        tab = self.tab_opt
        tab.columnconfigure(0, weight=1)

        sf = ttk.LabelFrame(tab, text="Objective & budget", padding=12)
        sf.grid(row=0, column=0, sticky="new")
        self.lam_cost_var = tk.StringVar(value="0.05")
        self.lam_mass_var = tk.StringVar(value="0.02")
        self.mass_budget_var = tk.StringVar(value="")
        ttk.Label(sf, text="λ cost :").grid(row=0, column=0, sticky="w")
        ttk.Entry(sf, textvariable=self.lam_cost_var, width=8).grid(row=0, column=1, padx=8, sticky="w")
        ttk.Label(sf, text="λ mass :").grid(row=0, column=2, sticky="w", padx=(16, 0))
        ttk.Entry(sf, textvariable=self.lam_mass_var, width=8).grid(row=0, column=3, padx=8, sticky="w")
        ttk.Label(sf, text="mass budget (optional) :").grid(row=0, column=4, sticky="w", padx=(16, 0))
        ttk.Entry(sf, textvariable=self.mass_budget_var, width=8).grid(row=0, column=5, padx=8, sticky="w")
        ttk.Label(sf, text="F = Q - λ_cost*cost - λ_mass*mass  (Q = dissipated heat)",
                  style="Muted.TLabel").grid(row=1, column=0, columnspan=6, sticky="w", pady=(6, 0))

        mf = ttk.LabelFrame(tab, text="Multi-fidelity & DE budget", padding=12)
        mf.grid(row=1, column=0, sticky="new", pady=(10, 0))
        self.coarse_var = tk.StringVar(value="20")
        self.fine_var = tk.StringVar(value="50")
        self.maxiter_var = tk.StringVar(value="20")
        self.popsize_var = tk.StringVar(value="8")
        for c, (lbl, var) in enumerate([("coarse mesh", self.coarse_var),
                                        ("fine mesh", self.fine_var),
                                        ("maxiter (DE)", self.maxiter_var),
                                        ("popsize (DE)", self.popsize_var)]):
            ttk.Label(mf, text=lbl + " :").grid(row=0, column=2 * c, sticky="w", padx=(0 if c == 0 else 12, 0))
            ttk.Entry(mf, textvariable=var, width=6).grid(row=0, column=2 * c + 1, padx=6, sticky="w")
        ttk.Label(mf, text="Coarse-mesh DE -> Nelder-Mead refinement (materials frozen) on the fine mesh.",
                  style="Muted.TLabel").grid(row=1, column=0, columnspan=8, sticky="w", pady=(6, 0))
        self.method_var = tk.StringVar(value="Multi-fidelity (DE -> NM)")
        ttk.Label(mf, text="Method :").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(mf, textvariable=self.method_var, state="readonly", width=26,
                     values=["Multi-fidelity (DE -> NM)", "Nelder-Mead",
                             "Differential Evolution", "Adam -> L-BFGS",
                             "Bayesian opt."]).grid(
                     row=2, column=1, columnspan=3, sticky="w", padx=6, pady=(8, 0))
        ttk.Label(mf, text="Same methods as Part I. The single-fidelity methods run on the fine mesh; "
                  "materials relaxed + rounded.", style="Muted.TLabel").grid(
                  row=3, column=0, columnspan=8, sticky="w", pady=(4, 0))

        bf = ttk.Frame(tab)
        bf.grid(row=2, column=0, sticky="we", pady=(12, 6))
        self.btn_opt = ttk.Button(bf, text="Run optimization",
                                  style="Accent.TButton", command=self.run_optimization)
        self.btn_opt.pack(side=tk.LEFT)
        ttk.Button(bf, text="Load this design into \"Single evaluation\"",
                   command=self.load_best_into_eval).pack(side=tk.LEFT, padx=8)
        ttk.Button(bf, text="Show optimized T",
                   command=lambda: self._select_viz("T_opt")).pack(side=tk.LEFT)

        pf = ttk.LabelFrame(tab, text="Progress", padding=12)
        pf.grid(row=3, column=0, sticky="new", pady=(8, 0))
        self.progress = ttk.Progressbar(pf, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X)
        self.progress_label = ttk.Label(pf, text="—", style="Muted.TLabel")
        self.progress_label.pack(anchor="w", pady=(4, 0))

        rf = ttk.LabelFrame(tab, text="Best design", padding=12)
        rf.grid(row=4, column=0, sticky="nsew", pady=(8, 0))
        tab.rowconfigure(4, weight=1)
        self.opt_result_var = tk.StringVar(value="Waiting…")
        ttk.Label(rf, textvariable=self.opt_result_var, style="Result.TLabel",
                  wraplength=950, justify="left").pack(anchor="w")

    # =======================================================================
    # Tab 3 - Visualization
    # =======================================================================
    def _build_tab_viz(self):
        tab = self.tab_viz
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        ctrl = ttk.LabelFrame(tab, text="What to display?", padding=12)
        ctrl.grid(row=0, column=0, sticky="new")
        self.viz_kind = tk.StringVar(value="mesh")
        for i, (txt, val) in enumerate([
                ("Mesh (current design)", "mesh"),
                ("T field (single evaluation)", "T_current"),
                ("T field (optimized)", "T_opt")]):
            ttk.Radiobutton(ctrl, text=txt, value=val, variable=self.viz_kind).grid(
                row=0, column=i, sticky="w", padx=8)
        bb = ttk.Frame(ctrl)
        bb.grid(row=1, column=0, columnspan=3, sticky="we", pady=(8, 0))
        ttk.Button(bb, text="Refresh", style="Accent.TButton",
                   command=self.refresh_viz).pack(side=tk.LEFT)
        ttk.Button(bb, text="Save PNG…", command=self.save_viz).pack(side=tk.LEFT, padx=8)

        plot_frame = ttk.Frame(tab, style="Card.TFrame")
        plot_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor=C_CARD, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame).update()
        self._placeholder("Run a computation or an optimization, then click \"Refresh\".")

    def _placeholder(self, msg, color=C_MUTED):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
                color=color, fontsize=12)
        ax.axis("off")
        self.canvas.draw()

    def _select_viz(self, kind):
        self.viz_kind.set(kind)
        self.nb.select(self.tab_viz)
        self.refresh_viz()

    # =======================================================================
    # Input reading / validation
    # =======================================================================
    def _read_design(self):
        try:
            m = np.array([NAME_TO_IDX[cb.get()] for cb in self.mat_cbs])
            t = np.array([float(e.get()) for e in self.t_entries])
            l = np.array([float(e.get()) for e in self.l_entries])
            Bi = float(self.bi_var.get())
            ms = int(self.mesh_size_var.get())
        except (ValueError, KeyError):
            messagebox.showerror("Error", "Invalid input (numbers expected).")
            return None
        if not (np.all((t >= model.T_LO) & (t <= model.T_HI))
                and np.all((l >= model.L_LO) & (l <= model.L_HI))
                and model.BI_LO <= Bi <= model.BI_HI and ms >= 10):
            messagebox.showwarning(
                "Out of bounds",
                f"t∈[{model.T_LO},{model.T_HI}], l∈[{model.L_LO},{model.L_HI}], "
                f"Bi∈[{model.BI_LO},{model.BI_HI}], mesh_size≥10.")
            return None
        return m, t, l, Bi, ms

    # =======================================================================
    # Actions - single evaluation
    # =======================================================================
    def calc_point(self):
        d = self._read_design()
        if d is None:
            return
        self.eval_result_var.set("Computing…")
        self._set_status("Single evaluation: FreeFEM…")
        self._busy(True)
        threading.Thread(target=self._do_calc, args=(d,), daemon=True).start()

    def _do_calc(self, d):
        m, t, l, Bi, ms = d
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            t_out = CACHE / "T_current.dat"
            Q, J = run_solver_param(model.k_vector(m), t, l, Bi, mesh_size=ms,
                                   t_out=str(t_out))
            cost, mass = model.cost_and_mass(m, t, l)
            self.last_T_file = t_out
            self.last_mesh_path = ensure_mesh(ms, t, l)
            names = ", ".join(MAT_NAMES[i] for i in m)
            msg = (f"Q = {Q:.6f}   (dissipated heat - performance objective)\n"
                   f"J = {J:.6f}   (mean temperature - info)\n"
                   f"Cost = {cost:.4f}   |   Mass = {mass:.4f}\n"
                   f"Materials: {names}\n"
                   f"Bi = {Bi:.4f}   |   mesh_size = {ms}")
            self.root.after(0, lambda: self.eval_result_var.set(msg))
            self.root.after(0, lambda: self._set_status("Computation done."))
        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: self.eval_result_var.set(f"Error: {err[:300]}"))
            self.root.after(0, lambda: messagebox.showerror("FreeFEM error", err))
        finally:
            self.root.after(0, lambda: self._busy(False))

    def reset_design(self):
        for cb in self.mat_cbs:
            cb.set("Copper")
        for e in self.t_entries:
            e.delete(0, "end"); e.insert(0, str(model.T_DEFAULT))
        for e in self.l_entries:
            e.delete(0, "end"); e.insert(0, str(model.L_DEFAULT))
        self.bi_var.set("0.3")
        self.mesh_size_var.set("40")
        self._set_status("Default design restored.")

    # =======================================================================
    # Actions - optimization
    # =======================================================================
    def run_optimization(self):
        try:
            lam_cost = max(0.0, float(self.lam_cost_var.get()))
            lam_mass = max(0.0, float(self.lam_mass_var.get()))
            coarse = int(self.coarse_var.get()); fine = int(self.fine_var.get())
            maxiter = int(self.maxiter_var.get()); popsize = int(self.popsize_var.get())
            mb = self.mass_budget_var.get().strip()
            mass_budget = float(mb) if mb else None
            method = self.method_var.get()
        except ValueError:
            messagebox.showerror("Error", "Invalid optimization parameters.")
            return
        if not messagebox.askyesno(
                "Optimization",
                f"Method: {method}\nλ_cost={lam_cost}, λ_mass={lam_mass}, "
                f"mass_budget={mass_budget}\nmesh {coarse}→{fine}, "
                f"maxiter={maxiter}, popsize={popsize}\n\nRun?"):
            return
        self.opt_result_var.set("running…")
        self._set_status("Optimization running…")
        self._busy(True)
        self.progress["maximum"] = maxiter
        self.progress["value"] = 0
        self.progress_label.config(text=f"0 / {maxiter} generations (coarse DE)")
        threading.Thread(
            target=self._do_opt,
            args=(method, lam_cost, lam_mass, mass_budget, coarse, fine, maxiter, popsize),
            daemon=True).start()

    def _do_opt(self, method, lam_cost, lam_mass, mass_budget, coarse, fine, maxiter, popsize):
        def progress_cb(done, total):
            self.root.after(0, lambda: self._update_progress(done, total))
        try:
            if method.startswith("Multi"):
                glob, res = optimize.multifidelity(
                    lam_cost=lam_cost, lam_mass=lam_mass, mass_budget=mass_budget,
                    coarse_mesh=coarse, fine_mesh=fine, maxiter=maxiter,
                    popsize=popsize, seed=42, progress_cb=progress_cb)
                extra = (f"Coarse DE: {glob['n_eval']} evals, {glob['time']:.0f}s  ->  "
                         f"fine refinement: {res['n_eval']} evals, {res['time']:.0f}s")
                mesh_used = fine
            else:
                obj = ParametricObjective(lam_cost, lam_mass, mass_budget, mesh_size=fine)
                if method == "Differential Evolution":
                    res = optimize.run_differential_evolution(
                        obj, popsize=popsize, maxiter=maxiter, seed=42,
                        progress_cb=progress_cb)
                elif method == "Nelder-Mead":
                    res = optimize.run_nelder_mead(obj, maxiter=max(200, maxiter * popsize))
                elif method == "Bayesian opt.":
                    res = optimize.run_bayesian(obj, n_calls=max(60, maxiter * popsize),
                                                seed=42, progress_cb=progress_cb)
                else:  # Adam -> L-BFGS
                    res = optimize.run_adam_then_lbfgs(obj, n_adam=maxiter, maxiter_lbfgs=50)
                extra = f"{method}: {res['n_eval']} evals, {res['time']:.0f}s (mesh {fine})"
                mesh_used = fine
            self.root.after(0, lambda: self.progress_label.config(text="done"))
            best_x = np.asarray(res["best_x"])
            self.last_best_x = best_x
            # Export the optimized T field (fine mesh)
            m, t, l, Bi = model.unpack(best_x)
            RESULTS.mkdir(parents=True, exist_ok=True)
            t_opt = RESULTS / "T_optimized_param.dat"
            run_solver_param(model.k_vector(m), t, l, Bi, mesh_size=mesh_used,
                            t_out=str(t_opt))
            self.last_opt_T = t_opt
            self.last_opt_mesh = ensure_mesh(mesh_used, t, l)
            msg = (f"Method: {method}\n"
                   f"F* = {res['best_F']:.6f}\n"
                   f"Q = {res['best_Q']:.6f}   |   J = {res['best_J']:.6f}\n"
                   f"Cost = {res['best_cost']:.4f}   |   Mass = {res['best_mass']:.4f}\n"
                   f"{extra}\n\n"
                   + model.describe(best_x))
            self.root.after(0, lambda: self.opt_result_var.set(msg))
            self.root.after(0, lambda: self._set_status("Optimization done."))
            self.root.after(0, lambda: self._select_viz("T_opt"))
        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: self.opt_result_var.set(f"Error: {err[:300]}"))
            self.root.after(0, lambda: messagebox.showerror("Error", err))
        finally:
            self.root.after(0, lambda: self._busy(False))

    def _update_progress(self, done, total):
        self.progress["maximum"] = total
        self.progress["value"] = done
        self.progress_label.config(text=f"{done} / {total} generations (coarse DE)")

    def load_best_into_eval(self):
        if self.last_best_x is None:
            messagebox.showinfo("No design", "Run an optimization first.")
            return
        m, t, l, Bi = model.unpack(self.last_best_x)
        for i in range(model.N_FINS):
            self.mat_cbs[i].set(MAT_NAMES[m[i]])
            self.t_entries[i].delete(0, "end"); self.t_entries[i].insert(0, f"{t[i]:.4f}")
            self.l_entries[i].delete(0, "end"); self.l_entries[i].insert(0, f"{l[i]:.4f}")
        self.bi_var.set(f"{Bi:.4f}")
        self.nb.select(self.tab_eval)
        self._set_status("Best design loaded into \"Single evaluation\".")

    # =======================================================================
    # Visualization
    # =======================================================================
    def refresh_viz(self):
        kind = self.viz_kind.get()
        try:
            if kind == "mesh":
                self._viz_mesh()
            elif kind == "T_current":
                self._viz_T(self.last_T_file, self.last_mesh_path,
                            "T field (single evaluation)")
            elif kind == "T_opt":
                self._viz_T(self.last_opt_T, self.last_opt_mesh,
                            "T field (optimized design)")
            self._set_status(f"Visualization: {kind}")
        except FileNotFoundError as e:
            self._placeholder(str(e), color=C_ERR)
        except Exception as e:
            messagebox.showerror("Visualization", str(e))

    def _viz_mesh(self):
        mp = self.last_mesh_path or self.last_opt_mesh
        if mp is None or not Path(mp).exists():
            raise FileNotFoundError("No mesh - run a computation first.")
        verts, tris, edges = read_freefem_mesh(mp)
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        draw_mesh(ax, verts, tris, edges, show_labels=True)
        self.canvas.draw()

    def _viz_T(self, t_file, mesh_path, title):
        if t_file is None or not Path(t_file).exists():
            raise FileNotFoundError("T field unavailable - run the matching computation first.")
        x, y, T = read_temperature_field(t_file)
        tri = None
        if mesh_path is not None and Path(mesh_path).exists():
            try:
                _, triangles, _ = read_freefem_mesh(mesh_path)
                if len(triangles) and triangles[:, :3].max() < len(x):
                    tri = triangles
            except Exception:
                tri = None
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        tcf = draw_temperature(ax, x, y, T, triangles=tri,
                               title=f"{title}\nT ∈ [{T.min():.3f}, {T.max():.3f}]",
                               cmap="inferno")
        self.fig.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04, label="T")
        self.canvas.draw()

    def save_viz(self):
        path = filedialog.asksaveasfilename(
            title="Save figure", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if path:
            self.fig.savefig(path, dpi=200, bbox_inches="tight")
            self._set_status(f"Figure saved: {path}")

    # =======================================================================
    def _busy(self, busy):
        state = "disabled" if busy else "normal"
        for w in (self.btn_calc, self.btn_opt):
            w.config(state=state)
        self.root.config(cursor="watch" if busy else "")


if __name__ == "__main__":
    root = tk.Tk()
    EnrichedGUI(root)
    root.mainloop()
