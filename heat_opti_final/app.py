#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interface graphique HEAT-COND.

Onglets :
  1. Calcul ponctuel  : J pour (k1..k5, Bi).
  2. Optimisation     : 3 modes (rapide / normale / approfondie).
  3. Visualisation    : maillage + champ T (intégré matplotlib).
"""

import json
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.freefem_interface import (  # noqa: E402
    CACHE_DIR,
    ensure_mesh,
    read_freefem_mesh,
    read_temperature_field,
    run_solver,
)
from src.optimization import run_differential_evolution  # noqa: E402
from src.utils import RESULTS_DIR, load_best_design, save_history  # noqa: E402
from src.visualization import (  # noqa: E402
    draw_convergence,
    draw_mesh,
    draw_temperature,
    plot_convergence,
)

# ---------- Constantes ----------
MODES = {
    "Recherche rapide":      {"mesh_size": 25, "maxiter": 5,  "popsize": 4},
    "Recherche normale":     {"mesh_size": 50, "maxiter": 15, "popsize": 8},
    "Recherche approfondie": {"mesh_size": 80, "maxiter": 30, "popsize": 15},
}
BOUNDS = [(0.1, 1.0)] * 5 + [(0.01, 1.0)]
PARAM_LABELS = ["k1", "k2", "k3", "k4", "k5", "Bi"]
DEFAULT_PARAMS = {"k1": 0.5, "k2": 0.5, "k3": 0.5, "k4": 0.5, "k5": 0.5, "Bi": 0.5}
DEFAULT_MESH_SIZE = "50"
DEFAULT_MODE = "Recherche normale"
SETTINGS_PATH = Path(__file__).resolve().parent / "config" / "last_inputs.json"

# Palette
C_BG       = "#f4f6f8"
C_CARD     = "#ffffff"
C_PRIMARY  = "#2c3e50"
C_ACCENT   = "#2980b9"
C_OK       = "#27ae60"
C_ERR      = "#c0392b"
C_MUTED    = "#7f8c8d"


# ===========================================================================
class HeatCondGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("HEAT-COND — Plateforme d'optimisation")
        self.root.geometry("1080x780")
        self.root.minsize(960, 700)
        self.root.configure(bg=C_BG)

        self._setup_style()

        # État
        self.last_T_file: Path | None = None
        self.last_best_x: list | None = None
        self.last_history: list = []

        self._build_header()
        self._build_notebook()
        self._build_statusbar()

        # Pré-remplir depuis la dernière session si dispo
        self._load_settings()
        # Sauvegarder à la fermeture
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- Style ttk ----------
    def _setup_style(self) -> None:
        style = ttk.Style()
        # Theme 'clam' est très configurable et portable
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        base_font = tkfont.nametofont("TkDefaultFont")
        base_font.configure(size=10)
        self.root.option_add("*Font", base_font)

        style.configure(".", background=C_BG)
        style.configure("TFrame", background=C_BG)
        style.configure("Card.TFrame", background=C_CARD, relief="flat")
        style.configure("TLabel", background=C_BG, foreground=C_PRIMARY)
        style.configure("Card.TLabel", background=C_CARD, foreground=C_PRIMARY)
        style.configure("Muted.TLabel", background=C_BG, foreground=C_MUTED, font=("TkDefaultFont", 9))
        style.configure("CardMuted.TLabel", background=C_CARD, foreground=C_MUTED, font=("TkDefaultFont", 9))
        style.configure("Title.TLabel", background=C_PRIMARY, foreground="white",
                        font=("TkDefaultFont", 16, "bold"))
        style.configure("Subtitle.TLabel", background=C_PRIMARY, foreground="#bdc3c7",
                        font=("TkDefaultFont", 10))
        style.configure("Status.TLabel", background="#dfe4e8", foreground=C_PRIMARY,
                        padding=(8, 4))
        style.configure("Result.TLabel", background=C_CARD, foreground=C_PRIMARY,
                        font=("Courier", 10))
        style.configure("Header.TLabelframe", background=C_BG)
        style.configure("Header.TLabelframe.Label", background=C_BG, foreground=C_PRIMARY,
                        font=("TkDefaultFont", 10, "bold"))
        style.configure("TLabelframe", background=C_BG)
        style.configure("TLabelframe.Label", background=C_BG, foreground=C_PRIMARY,
                        font=("TkDefaultFont", 10, "bold"))
        style.configure("TNotebook", background=C_BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 8), font=("TkDefaultFont", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", C_CARD), ("!selected", "#dfe4e8")],
                  foreground=[("selected", C_PRIMARY), ("!selected", C_MUTED)])
        style.configure("Accent.TButton", background=C_ACCENT, foreground="white",
                        padding=(12, 6), font=("TkDefaultFont", 10, "bold"),
                        borderwidth=0)
        style.map("Accent.TButton",
                  background=[("active", C_PRIMARY), ("disabled", "#95a5a6")])
        style.configure("TButton", padding=(10, 5))
        style.configure("Horizontal.TProgressbar",
                        troughcolor="#dfe4e8", background=C_ACCENT, thickness=14)

    # ---------- Header ----------
    def _build_header(self) -> None:
        hdr = tk.Frame(self.root, bg=C_PRIMARY, height=72)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        inner = tk.Frame(hdr, bg=C_PRIMARY)
        inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=8)
        ttk.Label(inner, text="HEAT-COND Optimization Platform",
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(inner,
                  text="Conduction 2D stationnaire — FreeFEM++ + Python (SPEIT ONA Spring 2026)",
                  style="Subtitle.TLabel").pack(anchor="w")

    # ---------- Notebook ----------
    def _build_notebook(self) -> None:
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=12, pady=(10, 6))
        self.tab_eval = ttk.Frame(nb, padding=12)
        self.tab_opt = ttk.Frame(nb, padding=12)
        self.tab_viz = ttk.Frame(nb, padding=12)
        nb.add(self.tab_eval, text="  Calcul ponctuel  ")
        nb.add(self.tab_opt,  text="  Optimisation  ")
        nb.add(self.tab_viz,  text="  Visualisation  ")
        self._build_tab_eval()
        self._build_tab_opt()
        self._build_tab_viz()

    # ---------- Statusbar ----------
    def _build_statusbar(self) -> None:
        self.status_var = tk.StringVar(value="Prêt.")
        ttk.Label(self.root, textvariable=self.status_var,
                  style="Status.TLabel", anchor="w").pack(fill=tk.X, side=tk.BOTTOM)

    def _set_status(self, msg: str, kind: str = "info") -> None:
        self.status_var.set(msg)

    # =======================================================================
    # Onglet 1 — Calcul ponctuel
    # =======================================================================
    def _build_tab_eval(self) -> None:
        tab = self.tab_eval
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)

        # --- Paramètres ---
        pf = ttk.LabelFrame(tab, text="Paramètres physiques (x = k1..k5, Bi)", padding=12)
        pf.grid(row=0, column=0, sticky="new", padx=(0, 6))
        self.entries: dict[str, ttk.Entry] = {}
        for i, lbl in enumerate(PARAM_LABELS):
            low, high = BOUNDS[i]
            ttk.Label(pf, text=lbl + " :").grid(row=i, column=0, sticky="w", pady=3)
            e = ttk.Entry(pf, width=12)
            e.insert(0, str(DEFAULT_PARAMS[lbl]))
            e.grid(row=i, column=1, padx=8, pady=3, sticky="w")
            ttk.Label(pf, text=f"∈ [{low}, {high}]",
                      style="Muted.TLabel").grid(row=i, column=2, sticky="w")
            self.entries[lbl] = e

        # --- Maillage ---
        mf = ttk.LabelFrame(tab, text="Maillage", padding=12)
        mf.grid(row=0, column=1, sticky="new", padx=(6, 0))
        ttk.Label(mf, text="mesh_size :").grid(row=0, column=0, sticky="w", pady=3)
        self.mesh_size_var = tk.StringVar(value=DEFAULT_MESH_SIZE)
        ttk.Entry(mf, textvariable=self.mesh_size_var, width=8)\
            .grid(row=0, column=1, padx=8, pady=3, sticky="w")
        ttk.Label(mf, text="(entier ≥ 10)",
                  style="Muted.TLabel").grid(row=0, column=2, sticky="w")

        ttk.Label(mf, text="OU fichier .msh :")\
            .grid(row=1, column=0, sticky="w", pady=(10, 3))
        self.mesh_path_var = tk.StringVar(value="")
        ttk.Entry(mf, textvariable=self.mesh_path_var, width=28)\
            .grid(row=1, column=1, padx=8, pady=(10, 3), sticky="we")
        ttk.Button(mf, text="Parcourir…", command=self._browse_mesh)\
            .grid(row=2, column=1, padx=8, pady=2, sticky="w")
        ttk.Button(mf, text="Effacer", command=self._clear_mesh)\
            .grid(row=2, column=2, padx=4, pady=2, sticky="w")
        ttk.Label(mf, text="Si un .msh est fourni, mesh_size est ignoré.",
                  style="Muted.TLabel").grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))

        # --- Boutons ---
        bf = ttk.Frame(tab)
        bf.grid(row=1, column=0, columnspan=2, sticky="we", pady=(14, 6))
        self.btn_calc = ttk.Button(bf, text="Calculer J", style="Accent.TButton",
                                   command=self.calc_j)
        self.btn_calc.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_calc_show = ttk.Button(
            bf, text="Calculer J + afficher la solution",
            command=self.calc_j_and_show,
        )
        self.btn_calc_show.pack(side=tk.LEFT)
        ttk.Button(bf, text="↻ Reset", command=self.reset_inputs)\
            .pack(side=tk.RIGHT)
        ttk.Button(bf, text="★ Charger le meilleur design",
                   command=self.load_best_into_inputs)\
            .pack(side=tk.RIGHT, padx=(0, 8))

        # --- Résultat ---
        rf = ttk.LabelFrame(tab, text="Résultat", padding=12)
        rf.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        tab.rowconfigure(2, weight=1)
        self.eval_result_var = tk.StringVar(value="En attente…")
        ttk.Label(rf, textvariable=self.eval_result_var, style="Result.TLabel",
                  background=C_BG, wraplength=900, justify="left").pack(anchor="w")

    # =======================================================================
    # Onglet 2 — Optimisation
    # =======================================================================
    def _build_tab_opt(self) -> None:
        tab = self.tab_opt
        tab.columnconfigure(0, weight=1)

        # --- Mode ---
        mf = ttk.LabelFrame(tab, text="Mode de recherche", padding=12)
        mf.grid(row=0, column=0, sticky="new")
        self.mode_var = tk.StringVar(value=DEFAULT_MODE)
        ttk.Label(mf, text="Niveau :").grid(row=0, column=0, sticky="w")
        cb = ttk.Combobox(mf, textvariable=self.mode_var, state="readonly",
                          values=list(MODES.keys()), width=26)
        cb.grid(row=0, column=1, padx=8, sticky="w")
        cb.bind("<<ComboboxSelected>>", self._on_mode_change)
        self.mode_info = ttk.Label(mf, text="", style="Muted.TLabel")
        self.mode_info.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(mf, text="Algorithme : Differential Evolution (scipy)",
                  style="Muted.TLabel").grid(row=2, column=0, columnspan=3, sticky="w")
        self._on_mode_change()

        # --- Bouton ---
        bf = ttk.Frame(tab)
        bf.grid(row=1, column=0, sticky="we", pady=(14, 6))
        self.btn_opt = ttk.Button(bf, text="Lancer l'optimisation",
                                  style="Accent.TButton", command=self.run_opt)
        self.btn_opt.pack(side=tk.LEFT)
        self.btn_show_init_opt = ttk.Button(
            bf, text="Afficher T initial vs optimisé",
            command=lambda: self._select_viz_kind("compare"),
        )
        self.btn_show_init_opt.pack(side=tk.LEFT, padx=(8, 0))

        # --- Progression ---
        pf = ttk.LabelFrame(tab, text="Progression", padding=12)
        pf.grid(row=2, column=0, sticky="new", pady=(8, 0))
        self.progress = ttk.Progressbar(pf, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X)
        self.progress_label = ttk.Label(pf, text="—", style="Muted.TLabel")
        self.progress_label.pack(anchor="w", pady=(4, 0))

        # --- Résultat ---
        rf = ttk.LabelFrame(tab, text="Résultat", padding=12)
        rf.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        tab.rowconfigure(3, weight=1)
        self.opt_result_var = tk.StringVar(value="En attente…")
        ttk.Label(rf, textvariable=self.opt_result_var, style="Result.TLabel",
                  background=C_BG, wraplength=900, justify="left").pack(anchor="w")

    def _on_mode_change(self, *_):
        cfg = MODES[self.mode_var.get()]
        self.mode_info.config(
            text=f"mesh_size = {cfg['mesh_size']}   |   maxiter = {cfg['maxiter']}   |   popsize = {cfg['popsize']}"
        )

    # =======================================================================
    # Onglet 3 — Visualisation
    # =======================================================================
    def _build_tab_viz(self) -> None:
        tab = self.tab_viz
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        ctrl = ttk.LabelFrame(tab, text="Que visualiser ?", padding=12)
        ctrl.grid(row=0, column=0, sticky="new")
        self.viz_kind = tk.StringVar(value="mesh")
        kinds = [
            ("Maillage",                       "mesh"),
            ("Champ T (paramètres courants)",  "T_current"),
            ("T initial (de l'optimisation)",  "T_init"),
            ("T optimisé (de l'optimisation)", "T_opt"),
            ("T initial vs optimisé",          "compare"),
            ("Convergence (dernier run)",      "convergence"),
        ]
        for i, (txt, val) in enumerate(kinds):
            ttk.Radiobutton(ctrl, text=txt, value=val,
                            variable=self.viz_kind).grid(
                row=i // 3, column=i % 3, sticky="w", padx=8, pady=2)

        # Toggle d'échelle pour la vue compare
        self.shared_scale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            ctrl,
            text="Échelle de couleurs partagée (compare T_init / T_opt)",
            variable=self.shared_scale_var,
            command=self._on_shared_scale_change,
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(6, 0))

        btns = ttk.Frame(ctrl)
        btns.grid(row=3, column=0, columnspan=3, sticky="we", pady=(8, 0))
        ttk.Button(btns, text="Actualiser", style="Accent.TButton",
                   command=self.refresh_viz).pack(side=tk.LEFT)
        ttk.Button(btns, text="Sauvegarder PNG…",
                   command=self.save_viz).pack(side=tk.LEFT, padx=(8, 0))

        # --- Canvas matplotlib ---
        plot_frame = ttk.Frame(tab, style="Card.TFrame")
        plot_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor=C_CARD)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()
        self._draw_placeholder()

    def _on_shared_scale_change(self) -> None:
        # Si on est en train de regarder la comparaison, redessiner aussitôt.
        if self.viz_kind.get() == "compare":
            self.refresh_viz()

    def _draw_placeholder(self) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5,
                "Aucune visualisation chargée.\nLance un calcul ou clique sur « Actualiser ».",
                ha="center", va="center", transform=ax.transAxes,
                color=C_MUTED, fontsize=12)
        ax.axis("off")
        self.canvas.draw()

    def _select_viz_kind(self, kind: str) -> None:
        self.viz_kind.set(kind)
        # Aller sur l'onglet visualisation
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Notebook):
                child.select(self.tab_viz)
        self.refresh_viz()

    # =======================================================================
    # Validation des entrées
    # =======================================================================
    def _browse_mesh(self) -> None:
        path = filedialog.askopenfilename(
            title="Sélectionner un maillage FreeFEM",
            filetypes=[("Maillage FreeFEM", "*.msh"), ("Tous", "*.*")],
        )
        if path:
            self.mesh_path_var.set(path)

    def _clear_mesh(self) -> None:
        self.mesh_path_var.set("")

    def _get_params(self):
        try:
            vals = [float(self.entries[k].get()) for k in PARAM_LABELS]
        except ValueError:
            messagebox.showerror("Erreur", "Entrez des nombres valides.")
            return None
        if not all(0.1 <= v <= 1.0 for v in vals[:5]) or not (0.01 <= vals[5] <= 1.0):
            messagebox.showwarning("Hors borne",
                                   "k1..k5 ∈ [0.1, 1.0], Bi ∈ [0.01, 1.0]")
            return None
        return vals

    def _get_mesh_kwargs(self):
        if self.mesh_path_var.get().strip():
            p = Path(self.mesh_path_var.get().strip())
            if not p.exists():
                messagebox.showerror("Erreur", f"Maillage introuvable : {p}")
                return None
            return {"mesh_path": p}
        try:
            ms = int(self.mesh_size_var.get())
            if ms < 10:
                raise ValueError
        except ValueError:
            messagebox.showerror("Erreur", "mesh_size doit être un entier ≥ 10.")
            return None
        return {"mesh_size": ms}

    def _resolve_mesh_path(self, mesh_kw: dict) -> Path:
        if "mesh_path" in mesh_kw:
            return mesh_kw["mesh_path"]
        return ensure_mesh(mesh_kw["mesh_size"])

    # =======================================================================
    # Actions — calcul ponctuel
    # =======================================================================
    def calc_j(self) -> None:
        self._save_settings()
        self._calc_j_common(show_after=False)

    def calc_j_and_show(self) -> None:
        self._save_settings()
        self._calc_j_common(show_after=True)

    def _calc_j_common(self, show_after: bool) -> None:
        params = self._get_params()
        if not params:
            return
        mesh_kw = self._get_mesh_kwargs()
        if mesh_kw is None:
            return
        self.eval_result_var.set("Calcul en cours…")
        self._set_status("Calcul ponctuel : exécution de FreeFEM…")
        self._set_busy(True)
        threading.Thread(target=self._do_calc, args=(params, mesh_kw, show_after),
                         daemon=True).start()

    def _do_calc(self, params, mesh_kw, show_after: bool):
        try:
            t_out = CACHE_DIR / "T_current.dat"
            t_out.parent.mkdir(parents=True, exist_ok=True)
            J = run_solver(params, doplot=0, t_out=str(t_out), **mesh_kw)
            self.last_T_file = t_out
            msg = (
                f"J = {J:.8f}\n"
                f"x = (k1={params[0]:.4f}, k2={params[1]:.4f}, k3={params[2]:.4f}, "
                f"k4={params[3]:.4f}, k5={params[4]:.4f}, Bi={params[5]:.4f})"
            )
            self.root.after(0, lambda: self.eval_result_var.set(msg))
            self.root.after(0, lambda: self._set_status("Calcul terminé."))
            if show_after:
                self.root.after(0, lambda: self._select_viz_kind("T_current"))
        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: self.eval_result_var.set(f"Erreur : {err[:300]}"))
            self.root.after(0, lambda: messagebox.showerror("Erreur FreeFEM", err))
            self.root.after(0, lambda: self._set_status("Erreur lors du calcul."))
        finally:
            self.root.after(0, lambda: self._set_busy(False))

    # =======================================================================
    # Actions — optimisation
    # =======================================================================
    def run_opt(self) -> None:
        self._save_settings()
        cfg = MODES[self.mode_var.get()]
        if not messagebox.askyesno(
            "Optimisation",
            f"Mode : {self.mode_var.get()}\n"
            f"mesh_size = {cfg['mesh_size']}, maxiter = {cfg['maxiter']}, popsize = {cfg['popsize']}\n\n"
            "Lancer ?",
        ):
            return
        self.opt_result_var.set(f"[{self.mode_var.get()}] en cours…")
        self._set_status("Optimisation en cours…")
        self._set_busy(True)
        self.progress["maximum"] = cfg["maxiter"]
        self.progress["value"] = 0
        self.progress_label.config(text=f"0 / {cfg['maxiter']} générations")
        threading.Thread(target=self._do_opt, args=(cfg,), daemon=True).start()

    def _do_opt(self, cfg):
        def progress_cb(done, total):
            self.root.after(0, lambda: self._update_progress(done, total))
        try:
            ensure_mesh(cfg["mesh_size"])
            # T initial (x0 = 0.5)
            t_init = RESULTS_DIR / "T_initial.dat"
            t_init.parent.mkdir(parents=True, exist_ok=True)
            run_solver([0.5] * 5 + [0.5], mesh_size=cfg["mesh_size"],
                       doplot=0, t_out=str(t_init))
            # Optim
            res = run_differential_evolution(
                BOUNDS,
                maxiter=cfg["maxiter"], popsize=cfg["popsize"],
                mesh_size=cfg["mesh_size"], progress_cb=progress_cb,
            )
            self.last_history = res["history"]
            self.last_best_x = list(res["best_x"])
            save_history(res["history"])
            plot_convergence(res["history"])
            # T optimisé
            t_opt = RESULTS_DIR / "T_optimized.dat"
            run_solver(self.last_best_x, mesh_size=cfg["mesh_size"],
                       doplot=0, t_out=str(t_opt))

            msg = (
                f"Méthode : {res['method']}\n"
                f"Meilleur J = {res['best_J']:.8f}\n"
                f"Évaluations = {res['n_eval']}   |   Temps = {res['time']:.1f} s\n"
                f"x* = {[f'{v:.4f}' for v in res['best_x']]}\n\n"
                f"results/optimization_history.csv et convergence.png mis à jour.\n"
                f"results/T_initial.dat et T_optimized.dat exportés."
            )
            self.root.after(0, lambda: self.opt_result_var.set(msg))
            self.root.after(0, lambda: self._set_status("Optimisation terminée."))
            self.root.after(0, lambda: self._select_viz_kind("compare"))
        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: self.opt_result_var.set(f"Erreur : {err[:300]}"))
            self.root.after(0, lambda: messagebox.showerror("Erreur", err))
            self.root.after(0, lambda: self._set_status("Erreur lors de l'optimisation."))
        finally:
            self.root.after(0, lambda: self._set_busy(False))

    def _update_progress(self, done: int, total: int) -> None:
        self.progress["maximum"] = total
        self.progress["value"] = done
        self.progress_label.config(text=f"{done} / {total} générations")

    # =======================================================================
    # Visualisation
    # =======================================================================
    def refresh_viz(self) -> None:
        kind = self.viz_kind.get()
        try:
            if kind == "mesh":
                self._viz_mesh()
            elif kind == "T_current":
                self._viz_T_current()
            elif kind == "T_init":
                self._viz_T_file(RESULTS_DIR / "T_initial.dat", "T initial (x = 0.5)")
            elif kind == "T_opt":
                self._viz_T_file(RESULTS_DIR / "T_optimized.dat", "T optimisé")
            elif kind == "compare":
                self._viz_compare()
            elif kind == "convergence":
                self._viz_convergence()
            self._set_status(f"Visualisation : {kind}")
        except FileNotFoundError as e:
            self._draw_placeholder_msg(str(e))
            self._set_status("Fichier manquant.")
        except Exception as e:
            messagebox.showerror("Visualisation", str(e))
            self._set_status("Erreur de visualisation.")

    def _draw_placeholder_msg(self, msg: str) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, color=C_ERR, fontsize=11)
        ax.axis("off")
        self.canvas.draw()

    def _viz_mesh(self) -> None:
        mesh_kw = self._get_mesh_kwargs()
        if mesh_kw is None:
            return
        mesh_path = self._resolve_mesh_path(mesh_kw)
        vertices, triangles, edges = read_freefem_mesh(mesh_path)
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        draw_mesh(ax, vertices, triangles, edges, show_labels=True)
        self.fig.tight_layout()
        self.canvas.draw()

    def _viz_T_current(self) -> None:
        if self.last_T_file is None or not Path(self.last_T_file).exists():
            raise FileNotFoundError(
                "Aucune solution courante. Lance d'abord « Calculer J + afficher »."
            )
        self._viz_T_file(self.last_T_file, "T (paramètres courants)")

    def _viz_T_file(self, t_file: Path, title: str) -> None:
        if not Path(t_file).exists():
            raise FileNotFoundError(f"Fichier manquant : {t_file}")
        x, y, T = read_temperature_field(t_file)
        # Auto-détecter le .msh qui a produit ce fichier T.
        tri = None
        mesh_path = self._find_mesh_matching_T(x, y)
        if mesh_path is not None:
            try:
                _, triangles, _ = read_freefem_mesh(mesh_path)
                if len(triangles) and triangles[:, :3].max() < len(x):
                    tri = triangles
            except Exception:
                tri = None
        title_with_range = f"{title}\nT ∈ [{T.min():.3f}, {T.max():.3f}]"
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        tcf = draw_temperature(ax, x, y, T, triangles=tri,
                               title=title_with_range, cmap="inferno")
        self.fig.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04, label="T")
        self.fig.tight_layout()
        self.canvas.draw()

    # =======================================================================
    # Mesh-T matching (auto-détection)
    # =======================================================================
    def _find_mesh_matching_T(self, x_dat, y_dat):
        """Cherche dans cache/ un .msh dont les sommets matchent un fichier T.

        Préfère un match exact des coordonnées ; à défaut, un match par
        nombre de sommets (cas où plusieurs maillages ont la même densité).
        """
        x_dat = np.asarray(x_dat)
        y_dat = np.asarray(y_dat)
        fallback = None
        for p in sorted(CACHE_DIR.glob("mesh_*.msh")):
            try:
                verts, _, _ = read_freefem_mesh(p)
            except Exception:
                continue
            if len(verts) != len(x_dat):
                continue
            if (np.allclose(verts[:, 0], x_dat, atol=1e-9)
                    and np.allclose(verts[:, 1], y_dat, atol=1e-9)):
                return p
            if fallback is None:
                fallback = p
        return fallback

    def _viz_compare(self) -> None:
        f_init = RESULTS_DIR / "T_initial.dat"
        f_opt = RESULTS_DIR / "T_optimized.dat"
        if not f_init.exists() or not f_opt.exists():
            raise FileNotFoundError(
                "T_initial.dat ou T_optimized.dat manquant — lance d'abord une optimisation."
            )
        x0, y0, T0 = read_temperature_field(f_init)
        x1, y1, T1 = read_temperature_field(f_opt)

        # Auto-détecter le .msh qui a produit T_initial.dat (mêmes (x, y))
        # pour éviter la Delaunay convexe (= hexagone qui ignore les vides).
        tri = None
        mesh_path = self._find_mesh_matching_T(x0, y0)
        if mesh_path is not None:
            try:
                _, triangles, _ = read_freefem_mesh(mesh_path)
                if (len(triangles)
                        and triangles[:, :3].max() < len(x0)
                        and triangles[:, :3].max() < len(x1)):
                    tri = triangles
            except Exception:
                tri = None

        title_init = f"Initial (x = 0.5)\nT ∈ [{T0.min():.3f}, {T0.max():.3f}]"
        title_opt  = f"Optimisé\nT ∈ [{T1.min():.3f}, {T1.max():.3f}]"

        self.fig.clear()
        if self.shared_scale_var.get():
            # Échelle partagée : 1 seule colorbar
            vmin = min(T0.min(), T1.min())
            vmax = max(T0.max(), T1.max())
            ax1 = self.fig.add_subplot(121)
            draw_temperature(ax1, x0, y0, T0, triangles=tri,
                             title=title_init, vmin=vmin, vmax=vmax)
            ax2 = self.fig.add_subplot(122)
            tcf = draw_temperature(ax2, x1, y1, T1, triangles=tri,
                                   title=title_opt, vmin=vmin, vmax=vmax)
            self.fig.colorbar(tcf, ax=[ax1, ax2],
                              fraction=0.04, pad=0.04, label="T")
            self.fig.suptitle("T : initial vs optimisé  (échelle partagée)")
        else:
            # Échelle indépendante : 1 colorbar par panneau
            ax1 = self.fig.add_subplot(121)
            tcf1 = draw_temperature(ax1, x0, y0, T0, triangles=tri,
                                    title=title_init)
            self.fig.colorbar(tcf1, ax=ax1, fraction=0.046, pad=0.04, label="T")
            ax2 = self.fig.add_subplot(122)
            tcf2 = draw_temperature(ax2, x1, y1, T1, triangles=tri,
                                    title=title_opt)
            self.fig.colorbar(tcf2, ax=ax2, fraction=0.046, pad=0.04, label="T")
            self.fig.suptitle("T : initial vs optimisé  (échelles indépendantes)")
        self.fig.tight_layout()
        self.canvas.draw()

    def _viz_convergence(self) -> None:
        history = self.last_history
        if not history:
            # Tente de relire le CSV
            import pandas as pd
            csv = RESULTS_DIR / "optimization_history.csv"
            if csv.exists():
                history = pd.read_csv(csv).to_dict("records")
            else:
                raise FileNotFoundError(
                    "Aucun historique disponible — lance d'abord une optimisation."
                )
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        draw_convergence(ax, history)
        self.fig.tight_layout()
        self.canvas.draw()

    def save_viz(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Sauvegarder la figure",
            defaultextension=".png",
            initialdir=str(RESULTS_DIR),
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
        )
        if path:
            self.fig.savefig(path, dpi=200, bbox_inches="tight")
            self._set_status(f"Figure sauvegardée : {path}")

    # =======================================================================
    # Helpers UI
    # =======================================================================
    def _set_busy(self, busy: bool) -> None:
        state = ("disabled" if busy else "normal")
        for w in (self.btn_calc, self.btn_calc_show, self.btn_opt):
            w.config(state=state)
        if not busy:
            self.root.config(cursor="")
        else:
            self.root.config(cursor="watch")


    # =======================================================================
    # Persistance des inputs (entre sessions)
    # =======================================================================
    def _save_settings(self) -> None:
        """Sauvegarde les valeurs courantes pour le prochain lancement."""
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "params":    {k: self.entries[k].get() for k in PARAM_LABELS},
                "mesh_size": self.mesh_size_var.get(),
                "mesh_path": self.mesh_path_var.get(),
                "mode":      self.mode_var.get(),
            }
            SETTINGS_PATH.write_text(json.dumps(data, indent=2))
        except Exception:
            pass  # silencieux : la persistance ne doit jamais bloquer l'app

    def _load_settings(self) -> None:
        """Pré-remplit les champs depuis le dernier état sauvegardé, si présent."""
        if not SETTINGS_PATH.exists():
            return
        try:
            data = json.loads(SETTINGS_PATH.read_text())
        except Exception:
            return
        for k, v in data.get("params", {}).items():
            if k in self.entries:
                self.entries[k].delete(0, "end")
                self.entries[k].insert(0, str(v))
        if "mesh_size" in data:
            self.mesh_size_var.set(str(data["mesh_size"]))
        if "mesh_path" in data:
            self.mesh_path_var.set(str(data["mesh_path"]))
        if "mode" in data and data["mode"] in MODES:
            self.mode_var.set(str(data["mode"]))
            self._on_mode_change()
        self._set_status("Champs pré-remplis depuis la dernière session.")

    def load_best_into_inputs(self) -> None:
        """Pré-remplit les champs k1..k5, Bi avec le meilleur design connu."""
        best = load_best_design()
        if best is None:
            messagebox.showinfo(
                "Pas de meilleur design",
                "Aucun historique d'optimisation trouvé "
                "(results/optimization_history.csv manquant ou vide).",
            )
            return
        for k, v in zip(PARAM_LABELS, best):
            self.entries[k].delete(0, "end")
            self.entries[k].insert(0, f"{v:.6f}")
        self._set_status("Meilleur design chargé dans les champs.")

    def reset_inputs(self) -> None:
        """Restaure les valeurs par défaut codées en dur."""
        for k in PARAM_LABELS:
            self.entries[k].delete(0, "end")
            self.entries[k].insert(0, str(DEFAULT_PARAMS[k]))
        self.mesh_size_var.set(DEFAULT_MESH_SIZE)
        self.mesh_path_var.set("")
        self.mode_var.set(DEFAULT_MODE)
        self._on_mode_change()
        self._set_status("Valeurs par défaut restaurées.")

    def _on_close(self) -> None:
        self._save_settings()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    HeatCondGUI(root)
    root.mainloop()
