from pyDOE3 import lhs, fullfact
from scipy.stats import qmc # para sobol, halton
import numpy as np

def _lhs_sampling(n_samples: int, l_bounds, u_bounds) -> np.ndarray:
    """Genera muestras usando Latin Hypercube Sampling con pyDOE3."""
    lb = np.array(l_bounds, dtype=float)
    ub = np.array(u_bounds, dtype=float)
    # Generar LHS normalizado en [0,1]
    norm = lhs(len(lb), samples=n_samples, criterion='maximin')
    # Escalar a rangos reales
    return lb + norm * (ub - lb)

def _full_factorial_sampling(levels: int, l_bounds, u_bounds) -> np.ndarray:
    """Genera muestras usando diseño factorial completo con pyDOE3."""
    lb = np.array(l_bounds, dtype=float)
    ub = np.array(u_bounds, dtype=float)
    # Factores para cada variable
    factors = [levels] * len(lb)
    # fullfact genera valores en 0..levels-1
    ff = fullfact(factors)
    # Escalar niveles a rangos reales
    return lb + ff * (ub - lb) / (levels - 1)

def _random_sampling(n_samples: int, l_bounds, u_bounds) -> np.ndarray:
    """Genera muestras usando muestreo aleatorio uniforme."""
    lb = np.array(l_bounds, dtype=float)
    ub = np.array(u_bounds, dtype=float)
    return np.random.uniform(low=lb, high=ub, size=(n_samples, len(lb)))

def _sobol_sampling(n_samples: int, l_bounds, u_bounds) -> np.ndarray:
    """Mantiene Sobol personalizado con scipy.stats.qmc."""
    lb = np.array(l_bounds, dtype=float)
    ub = np.array(u_bounds, dtype=float)
    sampler = qmc.Sobol(d=len(lb), scramble=True)
    if (n_samples & (n_samples - 1)) == 0:
        sample = sampler.random_base2(m=int(np.log2(n_samples)))
    else:
        sample = sampler.random(n=n_samples)
    return qmc.scale(sample, lb, ub)

def _halton_sampling(n_samples: int, l_bounds, u_bounds) -> np.ndarray:
    """Mantiene Halton personalizado con scipy.stats.qmc."""
    lb = np.array(l_bounds, dtype=float)
    ub = np.array(u_bounds, dtype=float)
    sampler = qmc.Halton(d=len(lb), scramble=True)
    sample = sampler.random(n=n_samples)
    return qmc.scale(sample, lb, ub)
