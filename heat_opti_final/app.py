#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface graphique pour la plateforme HEAT-COND.
Utilise les modules existants (src/).
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import subprocess
import sys
import os

# Ajouter le répertoire courant pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class HeatCondGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HEAT-COND Optimization Platform")
        self.root.geometry("500x550")
        self.root.resizable(False, False)
        self.create_widgets()

    def create_widgets(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="HEAT-COND Optimization Platform", font=("Arial",14,"bold")).pack(pady=10)
        ttk.Separator(main, orient='horizontal').pack(fill=tk.X, pady=5)

        # Paramètres
        pf = ttk.LabelFrame(main, text="Paramètres", padding=10)
        pf.pack(fill=tk.X, pady=5)
        self.entries = {}
        labels = ["k1","k2","k3","k4","k5","Bi"]
        defaults = [0.5,0.5,0.5,0.5,0.5,0.5]
        for i, (lbl, val) in enumerate(zip(labels, defaults)):
            ttk.Label(pf, text=lbl).grid(row=i, column=0, sticky=tk.W, pady=2)
            e = ttk.Entry(pf, width=15)
            e.insert(0, str(val))
            e.grid(row=i, column=1, padx=10, pady=2)
            self.entries[lbl] = e

        # Boutons
        bf = ttk.Frame(main)
        bf.pack(pady=10)
        ttk.Button(bf, text="Calculer J (ponctuel)", command=self.calc_j).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="Optimisation automatique", command=self.run_opt).pack(side=tk.LEFT, padx=5)

        # Résultat
        rf = ttk.LabelFrame(main, text="Résultat", padding=10)
        rf.pack(fill=tk.BOTH, expand=True, pady=5)
        self.result_var = tk.StringVar(value="En attente...")
        ttk.Label(rf, textvariable=self.result_var, font=("Courier",10), wraplength=450).pack()

        self.progress = ttk.Progressbar(main, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        self.progress.pack_forget()

    def get_params(self):
        try:
            vals = [float(self.entries[k].get()) for k in ["k1","k2","k3","k4","k5","Bi"]]
            if not all(0.1<=v<=1.0 for v in vals[:5]) or not (0.01<=vals[5]<=1.0):
                messagebox.showwarning("Hors borne", "k1..k5 ∈ [0.1,1.0], Bi ∈ [0.01,1.0]")
                return None
            return vals
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des nombres valides.")
            return None

    def calc_j(self):
        params = self.get_params()
        if not params:
            return
        self.result_var.set("Calcul en cours...")
        threading.Thread(target=self._run_calc, args=(params,), daemon=True).start()

    def _run_calc(self, params):
        try:
            # Utilisation directe des modules existants
            from src.freefem_interface import write_params, run_freefem, read_objective
            write_params(params)
            run_freefem(doplot=0, mesh_size=50)
            J = read_objective()
            self.root.after(0, lambda: self.result_var.set(f"J = {J:.8f}"))
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda: self.result_var.set(f"Erreur : {err_msg[:80]}"))
            self.root.after(0, lambda: messagebox.showerror("Erreur", err_msg))

    def run_opt(self):
        if not messagebox.askyesno("Optimisation", "Cela peut prendre plusieurs minutes. Continuer ?"):
            return
        self.progress.pack()
        self.progress.start(10)
        self.result_var.set("Optimisation en cours...")
        threading.Thread(target=self._run_opt, daemon=True).start()

    def _run_opt(self):
        try:
            # Lancer main.py dans un sous-processus (pour éviter les conflits)
            result = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
            if result.returncode == 0:
                self.root.after(0, lambda: self.result_var.set("Optimisation terminée. Voir results/"))
                self.root.after(0, lambda: messagebox.showinfo("Terminé", "Optimisation terminée.\nRésultats dans results/"))
            else:
                self.root.after(0, lambda: self.result_var.set("Erreur dans l'optimisation"))
                self.root.after(0, lambda: messagebox.showerror("Erreur", result.stderr[:500]))
        except Exception as e:
            self.root.after(0, lambda: self.result_var.set(f"Erreur : {str(e)[:80]}"))
            self.root.after(0, lambda: messagebox.showerror("Erreur", str(e)))
        finally:
            self.root.after(0, self.progress.stop)
            self.root.after(0, self.progress.pack_forget)

if __name__ == "__main__":
    root = tk.Tk()
    app = HeatCondGUI(root)
    root.mainloop()