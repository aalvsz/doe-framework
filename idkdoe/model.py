import pandas as pd
import pickle
import warnings
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
        self._validate_config()
        self.inputs_df = pd.DataFrame()
        self.outputs_df = pd.DataFrame()
        self._prepare_variables()

    def _validate_config(self):
        """Valida la configuración cargada y los intervalos de variables."""
        for section in ['model', 'analysis']:
            if section not in self.config:
                raise ValueError(f"Sección '{section}' no encontrada en el archivo YAML")
        if hasattr(self, 'bounds'):
            for name, (lb, ub) in zip(self.variable_names, self.bounds):
                if lb >= ub:
                    raise ValueError(f"Bounds inconsistente para variable '{name}': {lb} >= {ub}")

    def _prepare_variables(self):
        """Prepara las variables para el DOE."""
        self.variables_info = {}
        self.variable_names = []
        self.bounds = []
        self.var_types = []
        self.discrete_values = {}
        
        analysis_vars = self.config['analysis']['params']['variables']
        for var_name in analysis_vars:
            var_name = var_name.strip()
            if var_name not in self.config:
                warnings.warn(f"Variable '{var_name}' no definida en configuración, se omitirá.")
                continue
            var_cfg = self.config[var_name]
            self.variables_info[var_name] = var_cfg
            if 'value interval' in var_cfg:
                lb, ub = var_cfg['value interval']
                if lb >= ub:
                    raise ValueError(f"Intervalo no válido en '{var_name}': {lb} >= {ub}")
                self.variable_names.append(var_name)
                self.bounds.append((lb, ub))
                vtype = var_cfg.get('type', 'continuo').lower()
                if vtype not in ['continuo', 'discreto']:
                    warnings.warn(f"Tipo de variable '{vtype}' no reconocido para {var_name}. Usando 'continuo'.")
                    vtype = 'continuo'
                self.var_types.append(vtype)
                if vtype == 'discreto':
                    self.discrete_values[var_name] = list(range(int(lb), int(ub) + 1))

    def generate_samples(self, method: str = 'LHS', n_samples: int = 10, **kwargs) -> List[Dict]:
        """Genera muestras usando el método de DOE especificado con validación de bounds."""
        method = method.upper()
        l_bounds = [b[0] for b in self.bounds]
        u_bounds = [b[1] for b in self.bounds]
        for lb, ub, name in zip(l_bounds, u_bounds, self.variable_names):
            if lb >= ub:
                raise ValueError(f"Bounds inconsistente para '{name}': {lb} >= {ub}")
        if method == 'LHS':
            raw = _lhs_sampling(n_samples, l_bounds, u_bounds)
        elif method == 'RANDOM':
            raw = _random_sampling(n_samples, l_bounds, u_bounds)
        elif method == 'FULLFACTORIAL':
            levels = kwargs.get('levels', 2)
            raw = _full_factorial_sampling(levels, l_bounds, u_bounds)
        elif method == 'SOBOL':
            raw = _sobol_sampling(n_samples, l_bounds, u_bounds)
        elif method == 'HALTON':
            raw = _halton_sampling(n_samples, l_bounds, u_bounds)
        else:
            raise ValueError(f"Método de muestreo '{method}' no reconocido")
        samples = []
        for sample in raw:
            rec = {}
            for i, name in enumerate(self.variable_names):
                val = sample[i]
                rec[name] = int(round(val)) if self.var_types[i] == 'discreto' else float(val)
            samples.append(rec)
        return samples

    def run_simulations(self, samples: List[Dict], output_prefix: str = "results"):
        """Ejecuta las simulaciones para cada muestra y guarda los resultados."""

        if self.inputs_df.empty:
            self.inputs_df = pd.DataFrame(columns=self.variable_names)
        if self.outputs_df.empty:
            self.outputs_df = pd.DataFrame()

        for i, sample in enumerate(samples):
            print(f"Ejecutando simulación {i+1}/{len(samples)}...")
            try:
                #print(f"Entrada: {sample}")
                result     = self.model.idk_run(sample)
                #print(f"Salida: {result}")
                self.inputs_df = pd.concat([self.inputs_df, pd.DataFrame([sample])], ignore_index=True)
                self.outputs_df = pd.concat([self.outputs_df, pd.DataFrame([result])], ignore_index=True)
                _save_results(self, self.config['analysis']['params']['tracking']['path'], output_prefix)
            except Exception as e:
                print(f"Error en simulación {i+1}: {e}")

        """model_path = self.config['model']['pathModel']
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        if self.inputs_df.empty:
            self.inputs_df = pd.DataFrame(columns=self.variable_names)
        if self.outputs_df.empty:
            self.outputs_df = pd.DataFrame()
        for i, sample in enumerate(samples):
            print(f"Ejecutando simulación {i+1}/{len(samples)}...")
            try:
                print(f"Entrada: {sample}")
                out = model.idk_run(sample)
                print(f"Salida: {out}")
                self.inputs_df = pd.concat([self.inputs_df, pd.DataFrame([sample])], ignore_index=True)
                self.outputs_df = pd.concat([self.outputs_df, pd.DataFrame([out])], ignore_index=True)
                _save_results(self, self.config['analysis']['params']['tracking']['path'], output_prefix)
            except Exception as e:
                print(f"Error en simulación {i+1}: {e}")"""

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