import os
import subprocess
from dotenv import load_dotenv

load_dotenv()
FREEFEM_EXEC = os.getenv("FREEFEM_PATH")
if FREEFEM_EXEC is None:
    raise ValueError("FREEFEM_PATH non défini dans .env")

FREEFEM_SCRIPT = "scripts/heat_solver.edp"

def write_params(x, filename="params.txt"):
    with open(filename, "w") as f:
        for val in x:
            f.write(f"{val}\n")

def run_freefem(doplot=0, mesh_size=50):
    cmd = [FREEFEM_EXEC, FREEFEM_SCRIPT, "-doplot", str(doplot), "-meshsize", str(mesh_size)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError("FreeFEM execution failed")
    return result

def read_objective(filename="objective.txt"):
    with open(filename, "r") as f:
        return float(f.readline().strip())