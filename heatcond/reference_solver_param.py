"""Réplique NumPy pure du solveur paramétrique FreeFEM `freefem/solver_param.edp`.

Permet d'évaluer (Q, J) du modèle paramétrique — géométrie variable
(épaisseur t_i, longueur l_i) + conductivités k_i — SANS FreeFEM, afin de
rejouer la comparaison des optimiseurs de la Partie II.

Le domaine en peigne est une **union de rectangles** alignés sur les axes
(le tronc + une paire d'ailettes par étage). On construit donc un maillage
**rectilinéaire exact** : toutes les arêtes des rectangles sont des lignes de
grille, donc chaque cellule est entièrement dedans ou dehors — la frontière est
représentée exactement (pas d'effet d'escalier). Chaque cellule conservée est
découpée en deux triangles ; on assemble la formulation P1 identique à
`solver_param.edp` :

  - kappa = 1 dans le tronc, k_i dans l'ailette i (sinon 1) ;
  - terme de Robin  Bi * ∫_Γ2 T v  ;  Dirichlet T = 1 sur la base (label 1) ;
  - J = ∫_Γ2 T / |Γ2|  ;  Q = ∫_Γ2 Bi·T  (chaleur dissipée, performance).
"""

from collections import defaultdict

import numpy as np

# ---- constantes géométriques (identiques à solver_param.edp / mesh_param.edp) ----
WS = 0.10
XG = (1.0 - WS) / 2.0      # 0.45
XD = XG + WS               # 0.55
FY = np.array([0.16, 0.32, 0.48, 0.64, 0.80])
EPS = 1e-9


def _rectangles(t, l):
    """Liste des rectangles (x0, x1, y0, y1) : tronc + 5 ailettes."""
    rects = [(XG, XD, 0.0, 1.0)]                      # tronc
    for i in range(5):
        x0 = max(0.0, XG - float(l[i]))
        x1 = min(1.0, XD + float(l[i]))
        y0 = float(FY[i]) - float(t[i]) / 2.0
        y1 = float(FY[i]) + float(t[i]) / 2.0
        rects.append((x0, x1, y0, y1))
    return rects


def _refine(cuts, h):
    """Insère des points entre les coupures pour un pas <= h."""
    out = [cuts[0]]
    for a, b in zip(cuts[:-1], cuts[1:]):
        n = max(1, int(np.ceil((b - a) / h)))
        for j in range(1, n + 1):
            out.append(a + (b - a) * j / n)
    return np.array(sorted(set(np.round(out, 12))))


def build_comb_mesh(t, l, mesh_size):
    """Construit le maillage rectilinéaire exact du peigne paramétré par (t, l).

    Renvoie (V, Tr, region, E) :
      V      : (nv, 2) coordonnées des sommets
      Tr     : (nt, 3) triangles (indices 0-based)
      region : (nt,) 0 = tronc/défaut (kappa 1), i+1 = ailette i (kappa k_i)
      E      : (nbe, 3) arêtes de bord [v1, v2, label]  (1 = base, 2 = Robin)
    """
    rects = _rectangles(t, l)
    h = 1.0 / float(mesh_size)

    xcuts = [0.0, 1.0, XG, XD]
    ycuts = [0.0, 1.0]
    for (x0, x1, y0, y1) in rects:
        xcuts += [x0, x1]
        ycuts += [y0, y1]
    xcuts = sorted(set(min(max(c, 0.0), 1.0) for c in xcuts))
    ycuts = sorted(set(min(max(c, 0.0), 1.0) for c in ycuts))
    xs = _refine(xcuts, h)
    ys = _refine(ycuts, h)

    def in_dom(px, py):
        for (x0, x1, y0, y1) in rects:
            if x0 - EPS <= px <= x1 + EPS and y0 - EPS <= py <= y1 + EPS:
                return True
        return False

    def region_of(cx, cy):
        if XG <= cx <= XD:
            return 0
        for i in range(5):
            if abs(cy - FY[i]) <= t[i] / 2.0 and (cx <= XG or cx >= XD):
                return i + 1
        return 0

    nodes = []
    nid = {}

    def node(ix, iy):
        key = (ix, iy)
        idx = nid.get(key)
        if idx is None:
            idx = len(nodes)
            nid[key] = idx
            nodes.append((xs[ix], ys[iy]))
        return idx

    tris = []
    regions = []
    nx, ny = len(xs), len(ys)
    for ix in range(nx - 1):
        for iy in range(ny - 1):
            cx = 0.5 * (xs[ix] + xs[ix + 1])
            cy = 0.5 * (ys[iy] + ys[iy + 1])
            if not in_dom(cx, cy):
                continue
            n00 = node(ix, iy); n10 = node(ix + 1, iy)
            n11 = node(ix + 1, iy + 1); n01 = node(ix, iy + 1)
            reg = region_of(cx, cy)
            tris.append((n00, n10, n11)); regions.append(reg)
            tris.append((n00, n11, n01)); regions.append(reg)

    V = np.array(nodes, dtype=float)
    Tr = np.array(tris, dtype=int)
    region = np.array(regions, dtype=int)

    # arêtes de bord = arêtes n'appartenant qu'à un seul triangle
    edge_count = defaultdict(int)
    for a, b, c in Tr:
        for e in ((a, b), (b, c), (c, a)):
            edge_count[tuple(sorted(e))] += 1
    edges = []
    for (a, b), cnt in edge_count.items():
        if cnt != 1:
            continue
        ya, yb = V[a, 1], V[b, 1]
        xa, xb = V[a, 0], V[b, 0]
        base = (abs(ya) < EPS and abs(yb) < EPS
                and min(xa, xb) >= XG - EPS and max(xa, xb) <= XD + EPS)
        edges.append((a, b, 1 if base else 2))
    return V, Tr, region, np.array(edges, dtype=int)


_MESH_CACHE = {}


def _cached_mesh(t, l, mesh_size):
    """Cache des maillages par géométrie (clé arrondie). Évite de reconstruire le
    maillage à chaque évaluation quand la géométrie ne change pas (modèle simple)."""
    key = (int(mesh_size), tuple(np.round(np.asarray(t, float), 4)),
           tuple(np.round(np.asarray(l, float), 4)))
    m = _MESH_CACHE.get(key)
    if m is None:
        m = build_comb_mesh(t, l, mesh_size)
        if len(_MESH_CACHE) < 256:
            _MESH_CACHE[key] = m
    return m


def solve_param(k, t, l, Bi, mesh_size=25):
    """Résout le modèle paramétrique et renvoie (Q, J).

    Signature identique à :func:`heatcond.freefem.run_solver_param`, de sorte que
    les deux solveurs sont interchangeables dans :class:`ParametricObjective`.
    k, t, l : longueur 5 (par ailette) ; Bi : scalaire.
    """
    k = np.asarray(k, dtype=float)
    Bi = float(Bi)
    V, Tr, region, E = _cached_mesh(t, l, mesh_size)
    nv = V.shape[0]
    xy = V
    tris = Tr

    p = xy[tris]
    x1, y1 = p[:, 0, 0], p[:, 0, 1]
    x2, y2 = p[:, 1, 0], p[:, 1, 1]
    x3, y3 = p[:, 2, 0], p[:, 2, 1]
    b = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1)
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1)
    area = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    kappa = np.ones(len(region))
    for i in range(5):
        kappa[region == i + 1] = k[i]

    K = np.zeros((nv, nv))
    inv4A = 1.0 / (4.0 * area)
    for a in range(3):
        for bb in range(3):
            val = kappa * (b[:, a] * b[:, bb] + c[:, a] * c[:, bb]) * inv4A
            np.add.at(K, (tris[:, a], tris[:, bb]), val)

    e2 = E[E[:, 2] == 2][:, :2]
    a2 = xy[e2[:, 0]]
    b2 = xy[e2[:, 1]]
    L2 = np.sqrt(((a2 - b2) ** 2).sum(axis=1))
    if len(e2):
        coef = Bi * L2 / 6.0
        i0, i1 = e2[:, 0], e2[:, 1]
        np.add.at(K, (i0, i0), 2 * coef)
        np.add.at(K, (i1, i1), 2 * coef)
        np.add.at(K, (i0, i1), 1 * coef)
        np.add.at(K, (i1, i0), 1 * coef)

    dir_nodes = np.unique(E[E[:, 2] == 1][:, :2].reshape(-1))
    free = np.ones(nv, dtype=bool)
    free[dir_nodes] = False

    T = np.zeros(nv)
    T[dir_nodes] = 1.0
    rhs = -K @ T
    T[free] = np.linalg.solve(K[np.ix_(free, free)], rhs[free])

    Ta = T[e2[:, 0]]
    Tb = T[e2[:, 1]]
    intT = (L2 * (Ta + Tb) / 2.0).sum()
    lenfin = L2.sum()
    J = intT / lenfin
    Q = Bi * intT
    return float(Q), float(J)


def solve_param_field(k, t, l, Bi, mesh_size=50):
    """Comme :func:`solve_param` mais renvoie aussi le champ T et le maillage.

    Renvoie (V, Tr, T) : sommets (nv,2), triangles (nt,3), température (nv,).
    Utile pour tracer le champ de température.
    """
    k = np.asarray(k, dtype=float)
    Bi = float(Bi)
    V, Tr, region, E = _cached_mesh(t, l, mesh_size)
    nv = V.shape[0]
    p = V[Tr]
    x1, y1 = p[:, 0, 0], p[:, 0, 1]
    x2, y2 = p[:, 1, 0], p[:, 1, 1]
    x3, y3 = p[:, 2, 0], p[:, 2, 1]
    b = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1)
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1)
    area = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    kappa = np.ones(len(region))
    for i in range(5):
        kappa[region == i + 1] = k[i]
    K = np.zeros((nv, nv))
    inv4A = 1.0 / (4.0 * area)
    for a in range(3):
        for bb in range(3):
            val = kappa * (b[:, a] * b[:, bb] + c[:, a] * c[:, bb]) * inv4A
            np.add.at(K, (Tr[:, a], Tr[:, bb]), val)
    e2 = E[E[:, 2] == 2][:, :2]
    a2 = V[e2[:, 0]]; b2 = V[e2[:, 1]]
    L2 = np.sqrt(((a2 - b2) ** 2).sum(axis=1))
    if len(e2):
        coef = Bi * L2 / 6.0
        i0, i1 = e2[:, 0], e2[:, 1]
        np.add.at(K, (i0, i0), 2 * coef); np.add.at(K, (i1, i1), 2 * coef)
        np.add.at(K, (i0, i1), 1 * coef); np.add.at(K, (i1, i0), 1 * coef)
    dir_nodes = np.unique(E[E[:, 2] == 1][:, :2].reshape(-1))
    free = np.ones(nv, dtype=bool); free[dir_nodes] = False
    T = np.zeros(nv); T[dir_nodes] = 1.0
    rhs = -K @ T
    T[free] = np.linalg.solve(K[np.ix_(free, free)], rhs[free])
    return V, Tr, T


if __name__ == "__main__":
    # Validation : géométrie par défaut (~ modèle simple) -> J(coin) ~ 0.73
    t = [0.06] * 5
    l = [0.45] * 5
    for name, k, Bi in [
        ("uniforme 0.5", [0.5] * 5, 0.5),
        ("coin (k=1, Bi=0.01)", [1.0] * 5, 0.01),
        ("k=1, Bi=1.0", [1.0] * 5, 1.0),
    ]:
        Q, J = solve_param(k, t, l, Bi, mesh_size=50)
        print(f"  {name:22s} -> Q={Q:.5f}  J={J:.5f}")
