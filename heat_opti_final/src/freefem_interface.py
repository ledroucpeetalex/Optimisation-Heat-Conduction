"""Interface Python <-> FreeFEM++ pour HEAT-COND.

Architecture :
  - mesh.edp génère le maillage une seule fois par taille (mis en cache).
  - solver.edp prend les paramètres (k1..k5, Bi) via getARGV et écrit
    "Objective_J=<val>" sur stdout.

Plus aucun fichier texte intermédiaire pour les paramètres / l'objectif.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from dotenv import load_dotenv

# ---------- Configuration ----------
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MESH_SCRIPT = SCRIPTS_DIR / "mesh.edp"
SOLVER_SCRIPT = SCRIPTS_DIR / "solver.edp"
CACHE_DIR = PROJECT_ROOT / "cache"

FREEFEM_EXEC = os.getenv("FREEFEM_PATH")
if FREEFEM_EXEC is None:
    raise ValueError("FREEFEM_PATH non défini dans .env")

_OBJ_RE = re.compile(r"Objective_J=([-+0-9.eE]+)")


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _mesh_cache_path(mesh_size: int) -> Path:
    return CACHE_DIR / f"mesh_{mesh_size}.msh"


# ---------- Mesh generation ----------
def ensure_mesh(mesh_size: int = 50, force: bool = False) -> Path:
    """Génère le maillage si nécessaire et renvoie son chemin."""
    _ensure_cache_dir()
    path = _mesh_cache_path(mesh_size)
    if path.exists() and not force:
        return path
    cmd = [
        FREEFEM_EXEC, "-nw", str(MESH_SCRIPT),
        "-meshsize", str(int(mesh_size)),
        "-out", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not path.exists():
        raise RuntimeError(
            "Échec de la génération du maillage.\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return path


# ---------- Solver ----------
def run_solver(
    x: Sequence[float],
    mesh_size: int = 50,
    mesh_path: Optional[Path] = None,
    doplot: int = 0,
    t_out: Optional[str] = None,
) -> float:
    """Lance le solveur FreeFEM avec x = [k1..k5, Bi] et renvoie J.

    Si `t_out` est fourni, le champ T est exporté (x y T par sommet).
    """
    if len(x) != 6:
        raise ValueError(f"x doit avoir 6 éléments, reçu {len(x)}")

    if mesh_path is None:
        mesh_path = ensure_mesh(mesh_size)
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Maillage introuvable : {mesh_path}")

    k1, k2, k3, k4, k5, Bi = (float(v) for v in x)
    cmd = [FREEFEM_EXEC]
    if not doplot:
        cmd.append("-nw")
    cmd += [
        str(SOLVER_SCRIPT),
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

    m = _OBJ_RE.search(result.stdout)
    if not m:
        raise RuntimeError(
            "Impossible de parser l'objectif dans la sortie FreeFEM.\n"
            f"STDOUT:\n{result.stdout}"
        )
    return float(m.group(1))


def read_temperature_field(t_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relit un fichier T (x y T) écrit par solver.edp."""
    data = np.loadtxt(t_file, comments="#")
    return data[:, 0], data[:, 1], data[:, 2]


# ---------- Mesh reader (format FreeFEM .msh) ----------
def read_freefem_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lit un fichier .msh FreeFEM ASCII.

    Format :
      ligne 1 : nv nt nbe
      nv lignes  : x y label
      nt lignes  : v1 v2 v3 label   (indices 1-based)
      nbe lignes : v1 v2 label

    Renvoie (vertices, triangles, edges) :
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
