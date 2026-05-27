"""Interface CLI : calcule J pour un (k1..k5, Bi) saisi au clavier."""

from src.freefem_interface import run_solver


def get_user_input():
    print("\n=== Calcul de J pour des paramètres saisis ===")
    print("Bornes : k1..k5 ∈ [0.1, 1.0], Bi ∈ [0.01, 1.0]")
    try:
        k1 = float(input("k1 : "))
        k2 = float(input("k2 : "))
        k3 = float(input("k3 : "))
        k4 = float(input("k4 : "))
        k5 = float(input("k5 : "))
        Bi = float(input("Bi : "))
        return [k1, k2, k3, k4, k5, Bi]
    except ValueError:
        print("Erreur : entrez des nombres valides.")
        return None


def main():
    x = get_user_input()
    if x is None:
        return
    print("\nCalcul en cours...")
    try:
        J = run_solver(x, mesh_size=50, doplot=0)
        print("\n" + "=" * 40)
        print("RÉSULTAT")
        print("=" * 40)
        print(
            f"Paramètres : k1={x[0]:.4f}, k2={x[1]:.4f}, k3={x[2]:.4f}, "
            f"k4={x[3]:.4f}, k5={x[4]:.4f}, Bi={x[5]:.4f}"
        )
        print(f"J = {J:.8f}")
        print("=" * 40)
    except Exception as e:
        print(f"Erreur : {e}")


if __name__ == "__main__":
    main()
