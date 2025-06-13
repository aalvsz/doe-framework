import os
import csv
import time
import multiprocessing
import pandas as pd
import pickle
import warnings
import itertools
import numpy as np
from typing import Dict, List
from idkdoe.sampling_methods import (
    _full_factorial_sampling,
    _lhs_sampling,
    _random_sampling,
    _sobol_sampling,
    _halton_sampling
)
from idkdoe.utils import _visualize_samples
from joblib import Parallel, delayed

def _run_single_simulation(model_pickle, sample):
            import pickle
            import pandas as pd

            model = pickle.loads(model_pickle)
            try:
                result = model.idk_run(sample)
                if isinstance(result, pd.DataFrame):
                    row = result.to_dict(orient='records')[0]
                elif isinstance(result, dict):
                    row = result
                elif isinstance(result, tuple) and isinstance(result[0], dict):
                    row = result[0]
                else:
                    row = dict(result)
                return sample, row
            except Exception as e:
                print(f"Error en simulación con muestra {sample}: {e}")
                return sample, {}
            
class idkDOE:
    def __init__(self, config_dict: Dict, model):
        """Inicializa el proyecto idkDOE cargando el dict de configuración."""
        self.config = config_dict
        self.model = model
        self.inputs_df = pd.DataFrame()
        self.outputs_df = pd.DataFrame()
        self._validate_config()
        self._prepare_variables()

    def _append_row_to_csv(self, result_dict, save_path):
        """Escribe una fila al archivo CSV de resultados de forma dinámica."""
        outputs_csv = os.path.join(save_path, "DOE_outputs.csv")
        row_dict = result_dict.copy()

        # Si el archivo no existe, crear con encabezados
        if not os.path.exists(outputs_csv):
            with open(outputs_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row_dict.keys())
                writer.writeheader()
                writer.writerow(row_dict)
        else:
            with open(outputs_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row_dict.keys())
                writer.writerow(row_dict)

    def _validate_config(self):
        """Valida la configuración cargada y los intervalos de variables."""
        for section in ['model', 'analysis']:
            if section not in self.config:
                raise ValueError(f"Sección '{section}' no encontrada en el archivo YAML")

    def _prepare_variables(self):
        """Prepara las variables continuas y discretas para el DOE."""
        self.variables_info = {}
        self.variable_names = []
        self.bounds = []                # solo para continuas
        self.var_types = []
        self.discrete_values = {}       # para discretas

        analysis_vars = self.config['analysis']['params']['variables']
        for var_name in analysis_vars:
            var_name = var_name.strip()
            if var_name not in self.config:
                warnings.warn(f"Variable '{var_name}' no definida en configuración, se omitirá.")
                continue
            cfg = self.config[var_name]
            self.variables_info[var_name] = cfg
            vtype = cfg.get('type', 'continuo').lower()
            self.variable_names.append(var_name)
            self.var_types.append(vtype)

            interval = cfg.get('value interval', [])
            if vtype == 'discreto':
                # Interpretar interval como lista de posibles valores
                if not isinstance(interval, list) or len(interval) == 0:
                    raise ValueError(f"Lista de valores no válida para variable discreta '{var_name}'")
                self.discrete_values[var_name] = interval
                # placeholder in bounds to preserve indexing
                self.bounds.append((None, None))
            else:
                # continuo: interval debe ser [lb, ub]
                if (not isinstance(interval, list) or len(interval) != 2):
                    raise ValueError(f"Intervalo no válido para variable continua '{var_name}'")
                lb, ub = interval
                if lb >= ub:
                    raise ValueError(f"Intervalo no válido en '{var_name}': {lb} >= {ub}")
                self.bounds.append((lb, ub))

    def generate_samples(self, method: str = 'LHS', n_samples: int = 10, **kwargs) -> List[Dict]:
        """Genera muestras continuas y discretas combinadas según el método DOE."""
        method = method.upper()

        # índices
        cont_idx = [i for i, t in enumerate(self.var_types) if t == 'continuo']
        disc_idx = [i for i, t in enumerate(self.var_types) if t == 'discreto']

        # preparar muestras continuas
        cont_bounds = [self.bounds[i] for i in cont_idx]
        cont_samples = []
        if cont_idx:
            l_bounds = [b[0] for b in cont_bounds]
            u_bounds = [b[1] for b in cont_bounds]
            if method == 'LHS':
                raw = _lhs_sampling(n_samples, l_bounds, u_bounds)
            elif method == 'RANDOM':
                raw = _random_sampling(n_samples, l_bounds, u_bounds)
            elif method == 'SOBOL':
                raw = _sobol_sampling(n_samples, l_bounds, u_bounds)
            elif method == 'HALTON':
                raw = _halton_sampling(n_samples, l_bounds, u_bounds)
            elif method == 'FULLFACTORIAL':
                # continuo: generar niveles
                levels = kwargs.get('levels', 2)
                cont_lists = [np.linspace(lb, ub, levels).tolist() for lb, ub in cont_bounds]
                raw = list(itertools.product(*cont_lists))
            else:
                raise ValueError(f"Método de muestreo '{method}' no reconocido")
            cont_samples = list(raw)
        else:
            # sin continuas, crear lista vacía de tuplas
            cont_samples = [()] * (n_samples if method != 'FULLFACTORIAL' else 1)

        # preparar muestras discretas
        disc_lists = [self.discrete_values[self.variable_names[i]] for i in disc_idx]
        if disc_idx:
            if method == 'FULLFACTORIAL':
                # full factorial: combinar discretas con continuas
                all_samples = list(itertools.product(*[cont_lists if cont_idx else [()]] + disc_lists))
                # reestructurar: cada elemento es tuple(cont_vals..., disc_val)
                samples = []
                for tup in all_samples:
                    cont_vals = tup[0] if cont_idx else []
                    disc_vals = tup[1:]
                    rec = {}
                    # asignar continuas
                    for j, idx in enumerate(cont_idx):
                        rec[self.variable_names[idx]] = float(cont_vals[j])
                    # asignar discretas
                    for j, idx in enumerate(disc_idx):
                        rec[self.variable_names[idx]] = disc_vals[j]
                    samples.append(rec)
                return samples
            else:
                # para otros métodos, elegir aleatoriamente discretas n_samples
                import random
                disc_samples = [tuple(random.choice(vals) for vals in disc_lists) for _ in range(len(cont_samples))]
        else:
            disc_samples = [()] * len(cont_samples)

        # combinar cont y disc para métodos no factorial
        samples = []
        for i, cont_vals in enumerate(cont_samples):
            rec = {}
            # continuas
            for j, idx in enumerate(cont_idx):
                rec[self.variable_names[idx]] = float(cont_vals[j])
            # discretas
            for j, idx in enumerate(disc_idx):
                rec[self.variable_names[idx]] = disc_samples[i][j]
            samples.append(rec)

        # Guardar los samples generados en CSV
        n_configs = self.config['analysis']['params'].get('n_configs', 1)
        chunk_size = int(np.ceil(len(samples) / n_configs))

        for i in range(n_configs):
            chunk = samples[i * chunk_size: (i + 1) * chunk_size]
            chunk_df = pd.DataFrame(chunk)
            samples_path = self.config['analysis']['params']['tracking']['path']

            chunk_df.to_csv(f"{samples_path}/DOE_inputs_{method}_part{i+1}.csv", index=False)
            print(f"Muestras guardadas en: DOE_inputs_{method}_part{i+1}.csv")

        return samples

    
    def run_simulations(self, samples: List[Dict], output_prefix: str = "results", parallel: bool = False, n_workers=None):
        """Ejecuta las simulaciones en serie o en paralelo y guarda los resultados, incluyendo errores si ocurren."""
        if self.inputs_df.empty:
            self.inputs_df = pd.DataFrame(columns=self.variable_names)
        if self.outputs_df.empty:
            self.outputs_df = pd.DataFrame()

        start_time = time.perf_counter()

        results = []

        def safe_simulation(sample):
            try:
                result = self.model.idk_run(sample)
                if isinstance(result, pd.DataFrame):
                    row = result.to_dict(orient='records')[0]
                elif isinstance(result, dict):
                    row = result
                elif isinstance(result, tuple) and isinstance(result[0], dict):
                    row = result[0]
                else:
                    row = dict(result)
            except Exception as e:
                print(f"Error en simulación con muestra {sample}: {e}")
                row = {k: None for k in sample.keys()}
                row["error"] = str(e)
            return sample, row

        if parallel:
            print("Ejecutando simulaciones en paralelo con joblib...")
            model_pickle = pickle.dumps(self.model)

            def parallel_safe_simulation(sample):
                try:
                    model = pickle.loads(model_pickle)
                    result = model.idk_run(sample)
                    if isinstance(result, pd.DataFrame):
                        row = result.to_dict(orient='records')[0]
                    elif isinstance(result, dict):
                        row = result
                    elif isinstance(result, tuple) and isinstance(result[0], dict):
                        row = result[0]
                    else:
                        row = dict(result)
                except Exception as e:
                    print(f"Error en simulación con muestra {sample}: {e}")
                    row = {k: None for k in sample.keys()}
                    row["error"] = str(e)
                return sample, row

            raw_results = Parallel(n_jobs=n_workers or -1, backend="loky", verbose=10)(
                delayed(parallel_safe_simulation)(sample) for sample in samples
            )

            for i, (sample, row) in enumerate(raw_results, 1):
                print(f"Simulación {i}/{len(samples)} completada.")
                try:
                    self._append_row_to_csv(row, self.config['analysis']['params']['tracking']['path'])
                    results.append((sample, row))
                except Exception as e:
                    print(f"Error al guardar resultado de muestra {i}: {e}")

        else:
            for i, sample in enumerate(samples):
                print(f"Ejecutando simulación {i+1}/{len(samples)}...")
                _, row = safe_simulation(sample)
                try:
                    self._append_row_to_csv(row, self.config['analysis']['params']['tracking']['path'])
                    results.append((sample, row))
                except Exception as e:
                    print(f"Error al guardar resultado de muestra {i+1}: {e}")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        n_evaluations = len(samples)
        n_cpus = multiprocessing.cpu_count()
        n_processes = n_workers if parallel and n_workers is not None else (n_cpus if parallel else 1)
        exec_mode = "Paralelo" if parallel else "Secuencial"
        save_path = self.config['analysis']['params']['tracking']['path']

        summary_text = (
            f"Resumen de la simulación DOE\n"
            f"----------------------------\n"
            f"Tiempo transcurrido (segundos): {elapsed_time:.2f}\n"
            f"Número total de evaluaciones: {n_evaluations}\n"
            f"Modo de ejecución: {exec_mode}\n"
            f"Número de procesos paralelos usados: {n_processes}\n"
            f"Número total de CPUs disponibles: {n_cpus}\n"
            f"Ruta de guardado: {save_path}\n"
        )

        resumen_path = os.path.join(save_path, f"{output_prefix}_summary.txt")
        with open(resumen_path, 'w') as f:
            f.write(summary_text)

        print("\n" + summary_text)


    def run_doe_from_csv(
        self,
        input_csv: str,
        output_prefix: str = "results",
        parallel: bool = False,
        n_workers: int = None
    ):
        """Permite ejecutar simulaciones usando un archivo CSV específico de muestras."""
        samples_path = self.config['analysis']['params']['tracking']['path']
        full_path = os.path.join(samples_path, input_csv)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"No se encontró el archivo: {full_path}")

        samples_df = pd.read_csv(full_path)
        samples = samples_df.to_dict(orient='records')

        print(f"Ejecutando simulaciones desde archivo: {input_csv}")
        self.run_simulations(samples, output_prefix, parallel, n_workers)


    def run_doe(
        self,
        method: str = 'LHS',
        n_samples: int = 10,
        output_prefix: str = "results",
        parallel: bool = False,
        n_workers: int = None,
        **kwargs
    ):
        """Ejecuta todo el proceso de DOE con confirmación y visualización adecuada."""
        samples = self.generate_samples(method, n_samples, **kwargs)
        _visualize_samples(samples, self.variable_names, self.config['analysis']['params']['tracking']['path'])
        self.run_simulations(samples, output_prefix, parallel, n_workers)
        print("Simulaciones completadas. Resultados guardados.")
