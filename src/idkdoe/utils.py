import os
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix, parallel_coordinates
from sklearn.decomposition import PCA
import pandas as pd 
from typing import List, Dict

def _scale_samples(samples: List[Dict]) -> pd.DataFrame:
        """Escala cada variable a [0,1] para visualización."""
        df = pd.DataFrame(samples)
        mins = df.min()
        maxs = df.max()
        # Evitar división por cero
        ranges = maxs - mins
        for col in df.columns:
            if ranges[col] == 0:
                df[col] = 0.5  # constante en medio si no hay variación
            else:
                df[col] = (df[col] - mins[col]) / ranges[col]
        return df

def _visualize_samples(samples: List[Dict], variable_names: List[str], output_path: str):
    """Visualiza las muestras escaladas: scatter o coordenadas paralelas."""
    df_scaled = _scale_samples(samples)
    if df_scaled.shape[1] <= 6:
        scatter_matrix(df_scaled, diagonal='hist', alpha=0.7)
        plt.suptitle("Muestras propuestas (escaladas)")
        plt.show()
    else:
        df_plot = df_scaled.copy()
        df_plot['__id'] = df_plot.index.astype(str)
        plt.figure(figsize=(12, 6))
        parallel_coordinates(
            df_plot,
            class_column='__id',
            cols=variable_names,
            alpha=0.7,
            colormap=plt.get_cmap('tab20', len(df_plot))
        )
        plt.xticks(rotation=90)
        plt.title('Coordenadas Paralelas de muestras propuestas (escaladas)')
        plt.xlabel('Variables')
        plt.ylabel('Valor normalizado')
        plt.legend([], [])
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "parallel_coordinates_plot.png"), dpi=300)