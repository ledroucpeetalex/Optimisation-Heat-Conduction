"""Modèle paramétrique HEAT-COND : géométrie variable + choix de matériau.

Vecteur de design (16D), regroupé par type :

    x = [ m1..m5 | t1..t5 | l1..l5 | Bi ]

  - m_i : index matériau DISCRET dans MATERIALS (0..N_MAT-1) -> fixe k_i, prix, densité
  - t_i : épaisseur de l'ailette i (continu)
  - l_i : longueur (extension hors du spine) de l'ailette i (continu)
  - Bi  : nombre de Biot (continu)

L'objectif scalarisé (cf. :mod:`heatcond.objective`) est
    F(x) = Q(x) - lam_cost * Coût(x) - lam_mass * Masse(x)
avec Coût = Σ prix(m_i)·A_i, Masse = Σ densité(m_i)·A_i et A_i = 2·t_i·l_i
(les deux ailettes symétriques gauche/droite).
"""

import numpy as np

# --------------------------------------------------------------------------
# Catalogue de matériaux.
#   k       : conductivité normalisée, cohérente avec le problème HEAT-COND
#   price   : prix par unité de volume (proxy de coût matériau)
#   density : densité (proxy de masse)
# Valeurs illustratives, à recalibrer si besoin.
# --------------------------------------------------------------------------
MATERIALS = [
    {"name": "Copper",    "k": 1.00, "price": 9.0, "density": 9.0},
    {"name": "Aluminium", "k": 0.60, "price": 3.0, "density": 2.7},
    {"name": "Steel",     "k": 0.25, "price": 1.5, "density": 7.8},
    {"name": "Polymer",   "k": 0.05, "price": 0.3, "density": 1.2},
]
N_MAT = len(MATERIALS)
N_FINS = 5

# Bornes par groupe de variables
M_LO, M_HI = 0, N_MAT - 1      # index matériau (discret)
T_LO, T_HI = 0.02, 0.14        # épaisseur (< espacement 0.16 -> pas de chevauchement)
L_LO, L_HI = 0.10, 0.45        # longueur (0.45 -> la pointe atteint la paroi x=0/1)
BI_LO, BI_HI = 0.01, 1.0       # Biot

# Géométrie par défaut (= modèle simple) pour référence
T_DEFAULT = 0.06
L_DEFAULT = 0.45


def bounds():
    """Bornes scipy pour le vecteur de design 16D."""
    return (
        [(M_LO, M_HI)] * N_FINS
        + [(T_LO, T_HI)] * N_FINS
        + [(L_LO, L_HI)] * N_FINS
        + [(BI_LO, BI_HI)]
    )


def integrality():
    """Masque booléen scipy (`integrality`) : les 5 matériaux sont entiers."""
    return np.array([True] * N_FINS + [False] * (2 * N_FINS) + [False])


def unpack(x):
    """Décompose x -> (m[int], t, l, Bi). Les index matériaux sont arrondis/clippés."""
    x = np.asarray(x, dtype=float)
    m = np.clip(np.round(x[0:N_FINS]).astype(int), 0, N_MAT - 1)
    t = x[N_FINS:2 * N_FINS]
    l = x[2 * N_FINS:3 * N_FINS]
    Bi = float(x[3 * N_FINS])
    return m, t, l, Bi


def k_vector(m):
    """Conductivités correspondant aux index matériaux."""
    return np.array([MATERIALS[i]["k"] for i in m], dtype=float)


def fin_areas(t, l):
    """Surface 2D de chaque paire d'ailettes (gauche+droite) : A_i = 2·t_i·l_i."""
    return 2.0 * np.asarray(t, dtype=float) * np.asarray(l, dtype=float)


def cost_and_mass(m, t, l):
    """Renvoie (coût total, masse totale) pour le design."""
    A = fin_areas(t, l)
    price = np.array([MATERIALS[i]["price"] for i in m], dtype=float)
    dens = np.array([MATERIALS[i]["density"] for i in m], dtype=float)
    return float(np.sum(price * A)), float(np.sum(dens * A))


def describe(x):
    """Représentation lisible d'un design (pour logs / rapport)."""
    m, t, l, Bi = unpack(x)
    cost, mass = cost_and_mass(m, t, l)
    lines = ["Parametric design:"]
    for i in range(N_FINS):
        mat = MATERIALS[m[i]]
        lines.append(
            f"  Fin {i + 1}: {mat['name']:10s} (k={mat['k']:.2f})  "
            f"t={t[i]:.3f}  l={l[i]:.3f}"
        )
    lines.append(f"  Bi = {Bi:.4f}")
    lines.append(f"  Cost = {cost:.4f}   Mass = {mass:.4f}")
    return "\n".join(lines)
