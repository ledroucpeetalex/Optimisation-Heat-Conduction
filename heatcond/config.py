"""Configuration centrale : chemins du projet et accès à l'exécutable FreeFEM++.

Toutes les constantes de chemin sont dérivées de la racine du dépôt, de sorte
que le package fonctionne quel que soit le répertoire courant. L'exécutable
FreeFEM++ est lu paresseusement depuis la variable d'environnement
``FREEFEM_PATH`` (fichier ``.env``) : importer le package ne nécessite donc PAS
que FreeFEM soit installé — seul l'appel effectif à un solveur le requiert.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Arborescence du dépôt ------------------------------------------------
#   heatcond/config.py  ->  PACKAGE_DIR = heatcond/  ->  ROOT = racine du dépôt
PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent

# Charge .env depuis la racine du dépôt, quel que soit le répertoire courant
# (important quand on lance les notebooks depuis notebooks/).
load_dotenv(ROOT / ".env")

FREEFEM_DIR = ROOT / "freefem"          # sources .edp
RESULTS_DIR = ROOT / "results"          # figures + données générées
CACHE_DIR = ROOT / "cache"              # maillages mis en cache (.msh)

# Sous-dossiers de résultats par partie du rapport
RESULTS_PART1 = RESULTS_DIR / "part1"   # modèle simple (J)
RESULTS_PART2 = RESULTS_DIR / "part2"   # modèle paramétrique (matériau + géométrie)

# --- Scripts FreeFEM ------------------------------------------------------
# Modèle simple (géométrie fixe)
MESH_SCRIPT = FREEFEM_DIR / "mesh.edp"
SOLVER_SCRIPT = FREEFEM_DIR / "solver.edp"
# Modèle paramétrique (géométrie variable : épaisseur/longueur des ailettes)
MESH_PARAM_SCRIPT = FREEFEM_DIR / "mesh_param.edp"
SOLVER_PARAM_SCRIPT = FREEFEM_DIR / "solver_param.edp"


def get_freefem_exec() -> str:
    """Renvoie le chemin de l'exécutable FreeFem++ ou lève une erreur explicite.

    Lecture paresseuse : on ne vérifie ``FREEFEM_PATH`` qu'au moment où un solveur
    est réellement lancé, pas à l'import du package.
    """
    exe = os.getenv("FREEFEM_PATH")
    if not exe:
        raise RuntimeError(
            "FREEFEM_PATH is not set. Create a .env file in the project root "
            "containing FREEFEM_PATH=\"/absolute/path/to/FreeFem++\" (see README)."
        )
    return exe


def ensure_dirs() -> None:
    """Crée les dossiers de cache et de résultats si nécessaire."""
    for d in (CACHE_DIR, RESULTS_DIR, RESULTS_PART1, RESULTS_PART2):
        d.mkdir(parents=True, exist_ok=True)
