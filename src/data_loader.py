"""Carga genérica del dataset procesado y split X/y.

Agnóstico a los nombres de columnas de features: solo asume que existe
una columna target binaria identificada en ``config.TARGET_COLUMN``.
"""

from pathlib import Path

import pandas as pd

from src import config


def load_processed_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Carga el dataset ya preprocesado (imputado, con EDA aplicado).

    Parameters
    ----------
    path : str or Path, optional
        Ruta al archivo CSV. Si no se provee, usa ``config.PROCESSED_DATA_PATH``.

    Returns
    -------
    pd.DataFrame
        Dataset completo, incluyendo la columna target.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe todavía (ej. mientras se espera el dataset
        final de la compañera de equipo).
    """
    data_path = Path(path) if path is not None else config.PROCESSED_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset procesado en {data_path}. "
            "Usa src.synthetic_data.generate_synthetic_dataset() mientras llega "
            "el dataset final, o ajusta config.PROCESSED_DATA_PATH."
        )
    return pd.read_csv(data_path)


def split_features_target(
    df: pd.DataFrame, target_column: str | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa un DataFrame en features (X) y target (y).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo.
    target_column : str, optional
        Nombre de la columna target. Si no se provee, usa
        ``config.TARGET_COLUMN``.

    Returns
    -------
    X : pd.DataFrame
        Todas las columnas excepto el target.
    y : pd.Series
        Columna target.
    """
    target_col = target_column or config.TARGET_COLUMN
    if target_col not in df.columns:
        raise KeyError(
            f"La columna target '{target_col}' no existe en el dataset. "
            f"Columnas disponibles: {list(df.columns)}"
        )
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
