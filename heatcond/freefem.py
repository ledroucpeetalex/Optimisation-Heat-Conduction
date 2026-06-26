"""Interface Python <-> FreeFEM++ pour HEAT-COND.

Deux familles de fonctions, une par modèle :

* **modèle simple** (géométrie fixe) — :func:`ensure_mesh` génère le maillage une
  fois par densité (mis en cache), :func:`run_solver` lance ``solver.edp`` avec
  ``x = [k1..k5, Bi]`` et renvoie ``J`` ;
* **modèle paramétrique** (géométrie variable) — :func:`ensure_mesh_param`
  régénère le maillage pour chaque géométrie ``(t_i, l_i)`` (cache par hash), et
  :func:`run_solver_param` renvoie ``(Q, J)``.

Aucun fichier texte intermédiaire : les solveurs impriment ``Objective_J=`` (et
``Objective_Q=`` pour le modèle paramétrique) sur stdout.
"""

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from . import config

_OBJ_J_RE = re.compile(r"Objective_J=([-+0-9.eE]+)")
_OBJ_Q_RE = re.compile(r"Objective_Q=([-+0-9.eE]+)")


# ===========================================================================
# Modèle simple : géométrie fixe, x = [k1..k5, Bi]
# ===========================================================================
def _mesh_cache_path(mesh_size: int) -> Path:
    return config.CACHE_DIR / f"mesh_{int(mesh_size)}.msh"


def ensure_mesh(mesh_size: int = 50, force: bool = False) -> Path:
    """Génère (si besoin) le maillage du modèle simple et renvoie son chemin."""
    config.ensure_dirs()
    path = _mesh_cache_path(mesh_size)
    if path.exists() and not force:
        return path
    cmd = [config.get_freefem_exec(), "-nw", str(config.MESH_SCRIPT),
           "-meshsize", str(int(mesh_size)), "-out", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not path.exists():
        raise RuntimeError(
            "Échec de la génération du maillage.\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return path


def run_solver(
    x: Sequence[float],
    mesh_size: int = 50,
    mesh_path: Optional[Path] = None,
    doplot: int = 0,
    t_out: Optional[str] = None,
) -> float:
    """Lance ``solver.edp`` avec ``x = [k1..k5, Bi]`` et renvoie ``J``.

    Si ``t_out`` est fourni, le champ T est exporté (``x y T`` par sommet).
    """
    if len(x) != 6:
        raise ValueError(f"x doit avoir 6 éléments, reçu {len(x)}")

    if mesh_path is None:
        mesh_path = ensure_mesh(mesh_size)
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Maillage introuvable : {mesh_path}")

    k1, k2, k3, k4, k5, Bi = (float(v) for v in x)
    cmd = [config.get_freefem_exec()]
    if not doplot:
        cmd.append("-nw")
    cmd += [
        str(config.SOLVER_SCRIPT),
        "-meshfile", str(mesh_path),
        "-k1", f"{k1}", "-k2", f"{k2}", "-k3", f"{k3}",
        "-k4", f"{k4}", "-k5", f"{k5}", "-Bi", f"{Bi}",
        "-doplot", str(int(doplot)),
    ]
    if t_out:
        cmd += ["-Tout", str(t_out)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Échec de l'exécution du solveur FreeFEM.\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    m = _OBJ_J_RE.search(result.stdout)
    if not m:
        raise RuntimeError(
            "Impossible de parser l'objectif dans la sortie FreeFEM.\n"
            f"STDOUT:\n{result.stdout}"
        )
    return float(m.group(1))


# ===========================================================================
# Modèle paramétrique : géométrie variable, maillage régénéré par design
# ===========================================================================
def _geom_key(mesh_size: int, t: Sequence[float], l: Sequence[float]) -> str:
    """Hash court de la géométrie pour nommer le maillage en cache."""
    s = f"{int(mesh_size)}|" + "|".join(f"{v:.4f}" for v in list(t) + list(l))
    return hashlib.md5(s.encode()).hexdigest()[:12]


def ensure_mesh_param(mesh_size: int, t: Sequence[float], l: Sequence[float],
                      force: bool = False) -> Path:
    """Génère (si besoin) le maillage paramétrique pour la géométrie (t, l)."""
    config.ensure_dirs()
    path = config.CACHE_DIR / f"mesh_{int(mesh_size)}_{_geom_key(mesh_size, t, l)}.msh"
    if path.exists() and not force:
        return path
    cmd = [config.get_freefem_exec(), "-nw", str(config.MESH_PARAM_SCRIPT),
           "-meshsize", str(int(mesh_size)), "-out", str(path)]
    for i in range(5):
        cmd += [f"-t{i + 1}", f"{float(t[i])}", f"-l{i + 1}", f"{float(l[i])}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not path.exists():
        raise RuntimeError(
            "Échec de la génération du maillage paramétrique.\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return path


def run_solver_param(
    k: Sequence[float],
    t: Sequence[float],
    l: Sequence[float],
    Bi: float,
    mesh_size: int = 50,
    t_out: Optional[str] = None,
    doplot: int = 0,
) -> Tuple[float, float]:
    """Lance ``solver_param.edp`` et renvoie ``(Q, J)``.

    Q = chaleur dissipée (objectif de performance), J = température moyenne (info).
    ``k, t, l`` : longueur 5 (par ailette) ; ``Bi`` : scalaire.
    """
    mesh_path = ensure_mesh_param(mesh_size, t, l)
    cmd = [config.get_freefem_exec()]
    if not doplot:
        cmd.append("-nw")
    cmd += [str(config.SOLVER_PARAM_SCRIPT), "-meshfile", str(mesh_path),
            "-Bi", f"{float(Bi)}", "-doplot", str(int(doplot))]
    for i in range(5):
        cmd += [f"-k{i + 1}", f"{float(k[i])}", f"-t{i + 1}", f"{float(t[i])}"]
    if t_out:
        cmd += ["-Tout", str(t_out)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Échec de l'exécution du solveur FreeFEM (paramétrique).\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    mq = _OBJ_Q_RE.search(result.stdout)
    mj = _OBJ_J_RE.search(result.stdout)
    if not mq or not mj:
        raise RuntimeError(
            "Impossible de parser les objectifs.\n" f"STDOUT:\n{result.stdout}"
        )
    return float(mq.group(1)), float(mj.group(1))


# ===========================================================================
# Lecteurs de fichiers FreeFEM (communs aux deux modèles)
# ===========================================================================
def read_temperature_field(t_file) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relit un fichier T (``x y T``) écrit par un solveur."""
    data = np.loadtxt(t_file, comments="#")
    return data[:, 0], data[:, 1], data[:, 2]


def read_freefem_mesh(path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lit un fichier ``.msh`` FreeFEM ASCII.

    Renvoie ``(vertices, triangles, edges)`` :
      vertices  : (nv, 3) [x, y, label]
      triangles : (nt, 4) [v1, v2, v3, label]  -- indices 0-based
      edges     : (nbe, 3) [v1, v2, label]     -- indices 0-based
    """
    path = Path(path)
    with open(path) as f:
        tokens = f.read().split()
    it = iter(tokens)
    nv = int(next(it)); nt = int(next(it)); nbe = int(next(it))

    vertices = np.zeros((nv, 3))
    for i in range(nv):
        vertices[i, 0] = float(next(it))
        vertices[i, 1] = float(next(it))
        vertices[i, 2] = int(next(it))

    triangles = np.zeros((nt, 4), dtype=int)
    for i in range(nt):
        triangles[i, 0] = int(next(it)) - 1
        triangles[i, 1] = int(next(it)) - 1
        triangles[i, 2] = int(next(it)) - 1
        triangles[i, 3] = int(next(it))

    edges = np.zeros((nbe, 3), dtype=int)
    for i in range(nbe):
        edges[i, 0] = int(next(it)) - 1
        edges[i, 1] = int(next(it)) - 1
        edges[i, 2] = int(next(it))

    return vertices, triangles, edges
