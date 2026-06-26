"""Réplique Python (numpy pur) du solveur FreeFEM `freefem/solver.edp`.

But : pouvoir évaluer J(x) sans FreeFEM++ pour comparer les optimiseurs.
La formulation FEM P1 reproduit exactement solver.edp :
  - kappa par morceaux (spine=1, fin_i=k_i), évalué au centroïde de l'élément
  - terme de Robin  Bi * int_{label2} T v ds
  - Dirichlet T=1 sur label 1 (la base)
  - J = int_{label2} T ds / int_{label2} 1 ds
"""

import numpy as np

# ---- constantes géométriques (identiques à solver.edp / mesh.edp) ----
WS = 0.10
TF = 0.06
XG = (1.0 - WS) / 2.0     # 0.45
XD = XG + WS              # 0.55
FY = np.array([0.16, 0.32, 0.48, 0.64, 0.80])


def read_mesh(path):
    """Lit un .msh FreeFEM ASCII -> (vertices(nv,3), tris(nt,4 0-based), edges(nbe,3 0-based))."""
    with open(path) as f:
        tok = f.read().split()
    it = iter(tok)
    nv = int(next(it)); nt = int(next(it)); nbe = int(next(it))
    V = np.zeros((nv, 3))
    for i in range(nv):
        V[i, 0] = float(next(it)); V[i, 1] = float(next(it)); V[i, 2] = int(next(it))
    Tr = np.zeros((nt, 4), dtype=int)
    for i in range(nt):
        Tr[i, 0] = int(next(it)) - 1
        Tr[i, 1] = int(next(it)) - 1
        Tr[i, 2] = int(next(it)) - 1
        Tr[i, 3] = int(next(it))
    E = np.zeros((nbe, 3), dtype=int)
    for i in range(nbe):
        E[i, 0] = int(next(it)) - 1
        E[i, 1] = int(next(it)) - 1
        E[i, 2] = int(next(it))
    return V, Tr, E


class FEMProblem:
    """Pré-calcule toute la géométrie (indépendante de x) une seule fois."""

    def __init__(self, mesh_path):
        V, Tr, E = read_mesh(mesh_path)
        self.nv = V.shape[0]
        self.xy = V[:, :2]
        self.tris = Tr[:, :3]

        # --- pré-calcul par triangle : b, c, aire, region (1..5 ailette / 0 spine) ---
        p = self.xy[self.tris]                      # (nt,3,2)
        x1, y1 = p[:, 0, 0], p[:, 0, 1]
        x2, y2 = p[:, 1, 0], p[:, 1, 1]
        x3, y3 = p[:, 2, 0], p[:, 2, 1]
        self.b = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1)   # (nt,3)
        self.c = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1)   # (nt,3)
        self.area = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

        # centroïde -> indice de region pour kappa
        xc = (x1 + x2 + x3) / 3.0
        yc = (y1 + y2 + y3) / 3.0
        self.region = self._classify(xc, yc)        # 0 = spine, 1..5 = ailette i

        # --- arêtes de bord label 2 (Robin) ---
        self.e2 = E[E[:, 2] == 2][:, :2]
        a2 = self.xy[self.e2[:, 0]]; b2 = self.xy[self.e2[:, 1]]
        self.L2 = np.sqrt(((a2 - b2) ** 2).sum(axis=1))   # longueurs
        self.len_fin = self.L2.sum()                       # |Gamma_fin|

        # --- noeuds Dirichlet (label 1) ---
        e1 = E[E[:, 2] == 1][:, :2]
        self.dir_nodes = np.unique(e1.reshape(-1))

        # masque libre
        self.free = np.ones(self.nv, dtype=bool)
        self.free[self.dir_nodes] = False

    def _classify(self, xc, yc):
        reg = np.zeros(len(xc), dtype=int)           # défaut spine/0 -> kappa 1
        in_spine = (xc >= XG) & (xc <= XD)
        for i in range(5):
            in_fin = (np.abs(yc - FY[i]) <= TF / 2.0) & ((xc <= XG) | (xc >= XD))
            reg[(~in_spine) & in_fin] = i + 1
        # tout ce qui n'est ni spine ni fin reste 0 (kappa 1), comme solver.edp
        return reg

    def solve(self, x):
        """x = [k1..k5, Bi] -> J."""
        k = np.asarray(x[:5], dtype=float)
        Bi = float(x[5])

        # kappa par élément
        kappa = np.ones(len(self.region))
        for i in range(5):
            kappa[self.region == i + 1] = k[i]

        nv = self.nv
        K = np.zeros((nv, nv))

        # assemblage rigidité : Ke_ij = kappa*(bi bj + ci cj)/(4A)
        inv4A = 1.0 / (4.0 * self.area)
        for a in range(3):
            for bb in range(3):
                val = kappa * (self.b[:, a] * self.b[:, bb]
                               + self.c[:, a] * self.c[:, bb]) * inv4A
                np.add.at(K, (self.tris[:, a], self.tris[:, bb]), val)

        # terme de Robin sur label 2 : Me = Bi*L/6*[[2,1],[1,2]]
        if len(self.e2):
            coef = Bi * self.L2 / 6.0
            i0, i1 = self.e2[:, 0], self.e2[:, 1]
            np.add.at(K, (i0, i0), 2 * coef)
            np.add.at(K, (i1, i1), 2 * coef)
            np.add.at(K, (i0, i1), 1 * coef)
            np.add.at(K, (i1, i0), 1 * coef)

        # Dirichlet T=1 : élimination
        rhs = np.zeros(nv)
        T = np.zeros(nv)
        T[self.dir_nodes] = 1.0
        rhs = rhs - K @ T                       # déplace les colonnes Dirichlet
        free = self.free
        Kff = K[np.ix_(free, free)]
        rf = rhs[free]
        T[free] = np.linalg.solve(Kff, rf)

        # J = moyenne de T sur label 2
        Ta = T[self.e2[:, 0]]; Tb = T[self.e2[:, 1]]
        num = (self.L2 * (Ta + Tb) / 2.0).sum()
        return num / self.len_fin


if __name__ == "__main__":
    import sys
    mesh = sys.argv[1] if len(sys.argv) > 1 else "mesh_50.msh"
    prob = FEMProblem(mesh)
    print(f"mesh={mesh}  nv={prob.nv}  nt={len(prob.region)}  "
          f"n_edges_label2={len(prob.e2)}  n_dir_nodes={len(prob.dir_nodes)}")
    for name, x in [
        ("uniforme 0.5", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        ("coin (1..1, 0.01)", [1.0, 1.0, 1.0, 1.0, 1.0, 0.01]),
        ("NM design", [1, 1, 1, 1, 1, 0.01]),
    ]:
        print(f"  J({name:20s}) = {prob.solve(x):.6f}")
