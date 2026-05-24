from src.freefem_interface import write_params, run_freefem, read_objective

def get_user_input():
    print("\n=== Interface utilisateur : calcul de J pour des paramètres donnés ===")
    print("Veuillez entrer les valeurs suivantes (dans les bornes recommandées) :")
    try:
        k1 = float(input("k1 (0.1–1.0) : "))
        k2 = float(input("k2 (0.1–1.0) : "))
        k3 = float(input("k3 (0.1–1.0) : "))
        k4 = float(input("k4 (0.1–1.0) : "))
        k5 = float(input("k5 (0.1–1.0) : "))
        Bi = float(input("Bi (0.01–1.0) : "))
        return [k1, k2, k3, k4, k5, Bi]
    except ValueError:
        print("Erreur : veuillez entrer des nombres valides.")
        return None

def main():
    x = get_user_input()
    if x is None:
        return
    print("\nCalcul en cours...")
    write_params(x)
    try:
        run_freefem("-doplot", "0")
        J = read_objective()
        print("\n" + "="*40)
        print("RÉSULTAT")
        print("="*40)
        print(f"Paramètres : k1={x[0]:.4f}, k2={x[1]:.4f}, k3={x[2]:.4f}, k4={x[3]:.4f}, k5={x[4]:.4f}, Bi={x[5]:.4f}")
        print(f"Température moyenne sur les ailettes J = {J:.8f}")
        print("="*40)
    except Exception as e:
        print(f"Erreur lors de l'exécution : {e}")

if __name__ == "__main__":
    main()