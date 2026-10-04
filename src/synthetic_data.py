"""Generador de datos sintéticos con la misma forma del dataset real.

Permite desarrollar y probar todo el pipeline (preprocesamiento, balanceo,
optimización, validación anidada) mientras se espera el dataset
preprocesado definitivo de la compañera de equipo. Genera un target
binario desbalanceado (~78/22, igual que "Default of Credit Card
Clients") y features numéricas con nombres genéricos.

Las columnas categóricas nominales de ``config.CATEGORICAL_COLUMNS``
(confirmadas por el EDA real: ``SEX``/``EDUCATION``/``MARRIAGE``) se
generan con esos nombres exactos y una cardinalidad plausible, para poder
probar el encoding condicional de ``preprocessing.py`` end-to-end con
datos sintéticos. El resto de columnas son genéricas
(``feature_0``, ``feature_1``, ...) — el pipeline sigue sin acoplarse a
nombres de columnas NUMÉRICAS reales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src import config


def generate_synthetic_dataset(
    n_samples: int = config.SYNTHETIC_N_SAMPLES,
    n_features: int = config.SYNTHETIC_N_FEATURES,
    majority_ratio: float = config.SYNTHETIC_MAJORITY_RATIO,
    target_column: str | None = None,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Genera un dataset sintético binario y desbalanceado.

    Parameters
    ----------
    n_samples : int, default=config.SYNTHETIC_N_SAMPLES
        Número de observaciones a generar.
    n_features : int, default=config.SYNTHETIC_N_FEATURES
        Número de features numéricas a generar.
    majority_ratio : float, default=config.SYNTHETIC_MAJORITY_RATIO
        Proporción de la clase mayoritaria (0), en (0, 1).
    target_column : str, optional
        Nombre de la columna target. Si no se provee, usa
        ``config.TARGET_COLUMN``.
    random_state : int, default=config.RANDOM_STATE
        Semilla para reproducibilidad.

    Returns
    -------
    pd.DataFrame
        Dataset sintético con columnas ``feature_0..feature_{n-1}``
        (algunas renombradas a ``config.CATEGORICAL_COLUMNS``, ej.
        ``SEX``/``EDUCATION``/``MARRIAGE``) y la columna target binaria,
        con el mismo desbalance esperado en el dataset real.
    """
    target_col = target_column or config.TARGET_COLUMN
    minority_ratio = 1.0 - majority_ratio

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=max(0, n_features // 4),
        n_clusters_per_class=2,
        weights=[majority_ratio, minority_ratio],
        flip_y=0.02,
        class_sep=0.8,
        random_state=random_state,
    )

    rng = np.random.default_rng(random_state)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)

    # Simula tipos mixtos (algunas features categóricas discretizadas),
    # comunes en datasets financieros reales. Las primeras
    # len(config.CATEGORICAL_COLUMNS) se renombran a los nombres reales
    # confirmados por el EDA, con una cardinalidad plausible por variable;
    # cualquier columna categórica adicional simulada queda con nombre
    # genérico y 4 categorías por defecto.
    named_categoricals = list(config.CATEGORICAL_COLUMNS)
    cardinality_by_name = {"SEX": 2, "EDUCATION": 4, "MARRIAGE": 3}
    default_cardinality = 4

    n_categorical = max(len(named_categoricals), n_features // 5, 1)
    n_categorical = min(n_categorical, n_features)
    categorical_source_cols = list(rng.choice(feature_names, size=n_categorical, replace=False))

    rename_map: dict[str, str] = {}
    for i, source_col in enumerate(categorical_source_cols):
        target_name = named_categoricals[i] if i < len(named_categoricals) else source_col
        bins = cardinality_by_name.get(target_name, default_cardinality)
        # +1 para imitar la codificación 1-indexada típica de variables
        # categóricas en datasets financieros reales (ej. SEX: 1/2).
        df[source_col] = pd.cut(df[source_col], bins=bins, labels=False) + 1
        if target_name != source_col:
            rename_map[source_col] = target_name

    df = df.rename(columns=rename_map)
    df[target_col] = y
    return df
