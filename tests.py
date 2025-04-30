from idkdoe.model import idkDOE
import yaml

# Ejemplo de uso
if __name__ == "__main__":
    # 1) Leer el YAML
    with open(r'D:\idk_framework\idksimulation\doe_idkfem.yml', 'r') as f:
        data = yaml.safe_load(f)

    # Crear instancia de idkDOE
    doe = idkDOE(data)
    
    # Ejecutar DOE con diferentes métodos
    print("Generando muestras con LHS...")
    doe.run_doe(method="LHS", n_samples=50, output_prefix="doe_lhs_results", evaluate=False)
    
    print("\nGenerando muestras aleatorias...")
    doe.run_doe(method="RANDOM", n_samples=50, output_prefix="doe_random_results", evaluate=False)
    
    # DEMASIADO PESADO PARA AMS DE 5-6 VARIABLES
    """print("\nGenerando diseño factorial completo...") 
    doe.run_doe(method="FULLFACTORIAL", levels=3, output_prefix="doe_fullfactorial_results", evaluate=False)"""
    
    print("\nGenerando muestras con Sobol...")
    doe.run_doe(method="SOBOL", n_samples=64, output_prefix="doe_sobol_results", evaluate=False)  # Usar potencias de 2 para Sobol
    
    print("\nGenerando muestras con Halton...")
    doe.run_doe(method="HALTON", n_samples=50, output_prefix="doe_halton_results", evaluate=False)