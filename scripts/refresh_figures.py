"""Run both study notebooks and copy their figures into report/figures.

Executes the notebook code cells directly with Python (no Jupyter / nbconvert
needed): it only requires numpy, scipy, pandas and matplotlib. The notebooks use
SciPy optimizers on the pure NumPy reference solver, so FreeFEM is not required.

After this finishes, recompile the report:
    cd report && pdflatex main.tex && pdflatex main.tex

Usage (from anywhere):
    python scripts/refresh_figures.py
"""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    ROOT / "notebooks" / "part1_simple_objective.ipynb",
    ROOT / "notebooks" / "part2_material_geometry.ipynb",
]


def run_notebook(path):
    nb = json.load(open(path, encoding="utf-8"))
    codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    ns = {"__name__": "__main__"}
    cwd0 = os.getcwd()
    os.chdir(ROOT)
    try:
        for i, src in enumerate(codes):
            exec(compile(src, f"{path.name}:cell{i}", "exec"), ns)
    finally:
        os.chdir(cwd0)


def main():
    for nb in NOTEBOOKS:
        print(f"Running {nb.name} ...")
        run_notebook(nb)
    figdir = ROOT / "report" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    pngs = list((ROOT / "results" / "part1").glob("*.png"))
    pngs += list((ROOT / "results" / "part2").glob("*.png"))
    for p in pngs:
        shutil.copy(p, figdir / p.name)
    print(f"\nCopied {len(pngs)} figures into report/figures/.")
    print("Now recompile: cd report && pdflatex main.tex && pdflatex main.tex")
    print("(or tell Claude it is done, and it will recompile and update the report numbers).")


if __name__ == "__main__":
    main()
