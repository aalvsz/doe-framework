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
from idkdoe.utils import _save_results, _visualize_samples

class idkDOE:
    def __init__(self, config_dict: Dict, model):
        """Inicializa el proyecto idkDOE cargando el dict de configuración."""
        self.config = config_dict
        self.model = model
        self.inputs_df = pd.DataFrame()
        self.outputs_df = pd.DataFrame()
        self._validate_config()
        self._prepare_variables()

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

        return samples

    def run_simulations(self, samples: List[Dict], output_prefix: str = "results"):
        """Ejecuta las simulaciones para cada muestra y guarda los resultados."""
        # Inicialización de dataframes de IO
        if self.inputs_df.empty:
            self.inputs_df = pd.DataFrame(columns=self.variable_names)
        if self.outputs_df.empty:
            self.outputs_df = pd.DataFrame()

        output_cols = None  # guardará las columnas de salida esperadas
        for i, sample in enumerate(samples):
            print(f"Ejecutando simulación {i+1}/{len(samples)}...")
            try:
                result = self.model.idk_run(sample)
                # Convertir el resultado a dict si no lo es
                if isinstance(result, pd.DataFrame):
                    row = result.to_dict(orient='records')[0]
                elif isinstance(result, dict):
                    row = result
                elif isinstance(result, tuple) and isinstance(result[0], dict):
                    row = result[0]
                else:
                    # Si es otro tipo, intentar convertir
                    row = dict(result)
                # Capturar columnas de salida la primera vez
                if output_cols is None:
                    output_cols = list(row.keys())
                # Asegurar que row tenga todas las columnas
                for col in output_cols:
                    row.setdefault(col, 0)
            except Exception as e:
                print(f"Error en simulación {i+1}: {e}")
                # Si ya conocemos las columnas de salida, crear fila de ceros
                if output_cols is not None:
                    row = {col: 0 for col in output_cols}
                else:
                    # Columnas desconocidas: crear fila vacía
                    row = {}
            # Concatenar inputs y outputs
            self.inputs_df = pd.concat([self.inputs_df, pd.DataFrame([sample])], ignore_index=True)
            self.outputs_df = pd.concat([self.outputs_df, pd.DataFrame([row])], ignore_index=True)
            _save_results(self, self.config['analysis']['params']['tracking']['path'], output_prefix, self.inputs_df, self.outputs_df)

    def run_doe(
        self,
        method: str = 'LHS',
        n_samples: int = 10,
        output_prefix: str = "results",
        **kwargs
    ):
        """Ejecuta todo el proceso de DOE con confirmación y visualización adecuada."""
        evaluate = kwargs.pop('evaluate', True)
        while True:
            samples = self.generate_samples(method, n_samples, **kwargs)
            _visualize_samples(samples, self.variable_names)

            if not evaluate:
                print("Evaluación desactivada: finalizando sin simulaciones.")
                return
            choice = input("¿Evaluar en todos estos puntos? (Y/N): ")
            if choice.lower() == 'y':
                break
            if choice.lower() != 'n':
                print("Opción no válida. Ingresa 'Y' o 'N'.")
            else:
                print("Remuestreando...")
        self.run_simulations(samples, output_prefix)
        print("Simulaciones completadas. Resultados guardados.")
