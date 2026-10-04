"""Preprocesamiento: construcción del dataset final (limpieza + variables
derivadas, una sola vez, antes de cualquier split) y encoding + scaling
condicional según el modelo (dentro del pipeline de CV).

Dos etapas con responsabilidades distintas:

1. ``build_final_dataset`` — transformaciones **deterministas y sin
   parámetros aprendidos** (no dependen de qué filas queden en train o en
   validación, así que aplicarlas sobre el dataset completo no filtra
   información): deduplicación, reagrupación de categorías no
   documentadas, transformación log con signo de variables muy
   asimétricas, y reducción del bloque ``BILL_AMT1``-``BILL_AMT6``. Cada
   paso corresponde a un hallazgo del EDA (``notebooks/00_eda.ipynb``).
   Su resultado se guarda en ``config.PROCESSED_DATA_PATH``.
2. ``build_column_transformer`` / ``get_preprocessing_steps`` — one-hot de
   las categóricas y ``StandardScaler`` condicional. Estas SÍ aprenden
   parámetros (categorías, media/desviación), por eso viven dentro del
   pipeline de CV y se ajustan solo con el fold de entrenamiento.

El balanceo de clases (hallazgo de desbalance ~78/22) tampoco se aplica
aquí: se hace dentro del pipeline de CV (``balancing.py``) para que el
remuestreo nunca toque el fold de validación.

Todo el código del pipeline es agnóstico a los nombres de columnas
NUMÉRICAS: solo necesita saber cuáles son las CATEGÓRICAS nominales
(``config.CATEGORICAL_COLUMNS``) — cualquier otra columna se trata como
numérica automáticamente vía ``ColumnTransformer(remainder=...)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.models import ModelSpec

# --- Constantes del dataset UCI "Default of Credit Card Clients" ---
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]  # BILL_AMT1 = mes más reciente
PAY_AMOUNT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]
LOG_TRANSFORM_COLUMNS = ["LIMIT_BAL", *PAY_AMOUNT_COLUMNS]
LOG_SUFFIX = "_LOG"

# Códigos no documentados por la fuente (hallazgo del EDA) -> categoría "Otros".
UNDOCUMENTED_CATEGORY_MAP = {
    "EDUCATION": {0: 4, 5: 4, 6: 4},
    "MARRIAGE": {0: 3},
}

# Nombres de las variables derivadas del bloque BILL_AMT.
BILL_MEAN_COLUMN = "BILL_AMT_MEAN_6M_LOG"
BILL_TREND_COLUMN = "BILL_AMT_TREND_6M_LOG"
CREDIT_BALANCE_COLUMN = "N_CREDIT_BALANCE_MONTHS"


def signed_log1p(values):
    """``sign(x) * log(1 + |x|)``: log que tolera ceros y negativos.

    Se usa en vez de ``log1p`` plano porque ``BILL_AMT*`` tiene saldos
    negativos válidos (a favor del cliente) y ``log1p`` los destruiría;
    en vez de ``PowerTransformer`` porque no aprende parámetros, así que
    no hay nada que ajustar solo con el fold de entrenamiento.
    """
    return np.sign(values) * np.log1p(np.abs(values))


def load_raw_dataset(path=None) -> pd.DataFrame:
    """Lee el .xls original de UCI, renombra el target y descarta ``ID``.

    ``ID`` es un identificador único por construcción: dejarlo como
    columna lo convertiría en una feature más (ruido/fuga).
    """
    raw_path = path if path is not None else config.RAW_DATA_PATH
    df = pd.read_excel(raw_path, header=1)
    df = df.rename(columns={"default payment next month": config.TARGET_COLUMN})
    return df.drop(columns=["ID"], errors="ignore")


def regroup_undocumented_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Reagrupa los códigos no documentados de ``EDUCATION``/``MARRIAGE`` en "Otros".

    Returns
    -------
    df : pd.DataFrame
        Copia con las categorías reagrupadas.
    n_reassigned : dict[str, int]
        Filas reasignadas por columna (para el reporte).
    """
    out = df.copy()
    n_reassigned = {}
    for col, mapping in UNDOCUMENTED_CATEGORY_MAP.items():
        if col in out.columns:
            n_reassigned[col] = int(out[col].isin(list(mapping)).sum())
            out[col] = out[col].replace(mapping)
    return out, n_reassigned


def add_bill_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza ``BILL_AMT1``-``BILL_AMT6`` por 3 variables derivadas.

    El EDA midió VIF de hasta ~25 entre las seis facturas (cada mes es
    casi una copia del anterior) y SHAP les asigna importancia cercana a
    cero, así que se resumen en lo que sí distingue a un cliente de otro:

    - ``BILL_AMT_MEAN_6M_LOG``: nivel medio de deuda facturada (log con signo).
    - ``BILL_AMT_TREND_6M_LOG``: pendiente de la factura por mes (positiva =
      la deuda crece hacia el mes más reciente); casi no correlaciona con el
      nivel (~0.1), por eso no reintroduce multicolinealidad.
    - ``N_CREDIT_BALANCE_MONTHS``: cuántos de los 6 meses tuvieron factura
      negativa (saldo a favor del cliente), la "variable derivada" que el EDA
      dejó anotada para las facturas negativas.
    """
    out = df.copy()
    bills = out[BILL_COLUMNS].to_numpy(dtype=float)
    # tiempo creciente hacia el presente: BILL_AMT6 -> 0, ..., BILL_AMT1 -> 5
    t = np.arange(len(BILL_COLUMNS) - 1, -1, -1, dtype=float)
    t_centered = t - t.mean()
    slope = bills @ t_centered / (t_centered @ t_centered)

    out[BILL_MEAN_COLUMN] = signed_log1p(bills.mean(axis=1))
    out[BILL_TREND_COLUMN] = signed_log1p(slope)
    out[CREDIT_BALANCE_COLUMN] = (bills < 0).sum(axis=1).astype(int)
    return out.drop(columns=BILL_COLUMNS)


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Log con signo de ``LIMIT_BAL`` y ``PAY_AMT1``-``PAY_AMT6`` (renombradas con sufijo ``_LOG``).

    Asimetría original de hasta 30 (``PAY_AMT2``) con outliers de más de
    NT$1.6 millones; tras la transformación todas quedan por debajo de ~1.3
    en valor absoluto. Los árboles no cambian (solo dependen del orden),
    pero KNN/SVM/regresión logística/Naive Bayes dejan de estar dominados
    por unas pocas observaciones extremas.
    """
    out = df.copy()
    for col in LOG_TRANSFORM_COLUMNS:
        if col in out.columns:
            out[col + LOG_SUFFIX] = signed_log1p(out[col].astype(float))
            out = out.drop(columns=col)
    return out


def build_final_dataset(
    raw_df: pd.DataFrame, target_column: str | None = None
) -> tuple[pd.DataFrame, dict]:
    """Aplica TODO el preprocesamiento derivado de los hallazgos del EDA.

    Orden (cada paso sale de una sección de ``00_eda.ipynb``):

    1. Reagrupar categorías no documentadas de ``EDUCATION``/``MARRIAGE``.
    2. Eliminar filas duplicadas (sobre las columnas ya limpias y el target).
    3. Resumir ``BILL_AMT1``-``BILL_AMT6`` (multicolinealidad + facturas negativas).
    4. Log con signo de ``LIMIT_BAL``/``PAY_AMT*`` (asimetría/outliers extremos).

    Los pasos 3 y 4 son funciones fila a fila (no aprenden nada de otras
    filas), por eso es seguro aplicarlos sobre el dataset completo antes de
    particionar. La deduplicación sí mira otras filas, pero justamente por
    eso debe ocurrir una sola vez y antes de cualquier split.

    Returns
    -------
    final_df : pd.DataFrame
        Dataset listo para ``data_loader.split_features_target``.
    report : dict
        Conteos de cada paso, para registrarlos en el notebook/log.
    """
    target_col = target_column or config.TARGET_COLUMN
    df = raw_df.reset_index(drop=True)
    if "ID" in df.columns:
        df = df.drop(columns="ID")

    n_in = len(df)
    n_dups_raw = int(df.duplicated().sum())
    n_negative_bill_rows = int((df[BILL_COLUMNS] < 0).any(axis=1).sum())

    df, n_reassigned = regroup_undocumented_categories(df)
    df, n_removed = remove_duplicate_rows(df)
    n_conflicts = check_label_conflicts(df, target_column=target_col)
    df = add_bill_features(df)
    df = apply_log_transform(df)
    df = df[[c for c in df.columns if c != target_col] + [target_col]]  # target siempre al final

    report = {
        "filas_entrada": n_in,
        "duplicados_exactos_en_crudo": n_dups_raw,
        "filas_duplicadas_eliminadas": n_removed,
        "filas_salida": len(df),
        "filas_reasignadas_a_otros": n_reassigned,
        "filas_con_factura_negativa": n_negative_bill_rows,
        "grupos_con_etiqueta_conflictiva": n_conflicts,
        "columnas_salida": list(df.columns),
    }
    return df, report


def remove_duplicate_rows(
    df: pd.DataFrame, subset: list[str] | None = None
) -> tuple[pd.DataFrame, int]:
    """Elimina filas duplicadas ANTES de cualquier split train/test o CV.

    **Por qué debe hacerse una sola vez, antes de particionar:** si el
    dataset conserva filas duplicadas y luego se divide en folds (train/
    test o CV), un par de filas idénticas puede terminar una en
    entrenamiento y la otra en validación/test. El modelo entrena
    literalmente sobre esa fila y luego se evalúa sobre su copia exacta,
    inflando artificialmente la métrica de ese fold — una forma de
    leakage. El EDA identificó 35 filas así en el dataset original y las
    elimina (``00_eda.ipynb``); ``build_final_dataset`` aplica el mismo
    paso, así que el dataset de ``data/processed/`` ya llega sin
    duplicados (ver ``check_label_conflicts`` para el caso de duplicados
    con etiquetas distintas).

    Parameters
    ----------
    df : pd.DataFrame
    subset : list[str], optional
        Columnas a considerar para detectar duplicados. Por defecto
        (``None``) usa TODAS las columnas (incluido el target) — la
        definición más estricta de "fila duplicada". Pasar solo las
        columnas de features (sin el target) es más agresivo: verificar
        antes con ``check_label_conflicts`` si eso puede descartar filas
        con etiquetas distintas.

    Returns
    -------
    deduped : pd.DataFrame
        Sin duplicados, índice reseteado.
    n_removed : int
        Filas eliminadas (registrar en el reporte/log de experimentos).
    """
    deduped = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    n_removed = len(df) - len(deduped)
    return deduped, n_removed


def check_label_conflicts(
    df: pd.DataFrame,
    target_column: str | None = None,
    feature_columns: list[str] | None = None,
) -> int:
    """Cuenta grupos de filas con features idénticas pero target distinto.

    Un conflicto de etiqueta indica ruido irreducible en los datos (dos
    observaciones indistinguibles en las features disponibles, con
    resultados distintos). No se resuelve automáticamente aquí — solo se
    reporta, para dejarlo documentado como limitación del dataset (igual
    que el EDA documenta los códigos ambiguos -2/0 de ``PAY_x``).

    Parameters
    ----------
    target_column : str, optional
        Si no se provee, usa ``config.TARGET_COLUMN``.
    feature_columns : list[str], optional
        Si no se provee, usa todas las columnas de ``df`` excepto el target.

    Returns
    -------
    int
        Número de grupos de features duplicadas con más de un valor de target.
    """
    target_col = target_column or config.TARGET_COLUMN
    feature_cols = feature_columns or [c for c in df.columns if c != target_col]
    grouped = df.groupby(feature_cols, dropna=False)[target_col].nunique()
    return int((grouped > 1).sum())


def save_final_dataset(path=None) -> dict:
    """Regenera ``data/processed/dataset_final.csv`` desde el archivo original.

    Hace lo mismo que la Parte 5 de ``00_eda.ipynb``. Desde la raíz del
    proyecto: ``python -m src.preprocessing``.
    """
    out_path = path if path is not None else config.PROCESSED_DATA_PATH
    final_df, report = build_final_dataset(load_raw_dataset())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    return report


def build_column_transformer(
    categorical_columns: list[str] | None = None,
    apply_scaling: bool = False,
) -> ColumnTransformer:
    """Construye el ``ColumnTransformer`` de encoding + scaling condicional.

    - Columnas categóricas nominales -> ``OneHotEncoder``
      (``handle_unknown="ignore"`` para tolerar en un fold una categoría
      rara ausente de su set de entrenamiento; ``drop="if_binary"`` para
      no dejar una columna redundante en binarias como ``SEX``).
    - Todas las demás columnas (numéricas, sea cual sea su nombre) ->
      ``StandardScaler`` si ``apply_scaling=True`` (modelos sensibles a
      escala — ver ``ModelSpec.requires_scaling`` en ``models.py``: KNN,
      SVM, Regresión Logística), o sin transformar (passthrough) si no
      (árboles, Random Forest, XGBoost, Naive Bayes).

    Se usa como UN PASO MÁS del pipeline de CV (vía
    ``balancing.build_pipeline(preprocessing_steps=...)``), nunca aplicado
    por fuera de él, para que tanto el encoding como el scaling se ajusten
    ÚNICAMENTE sobre el fold de entrenamiento en cada iteración — igual
    que el balanceo (ver ``balancing.py``).

    Parameters
    ----------
    categorical_columns : list[str], optional
        Si no se provee, usa ``config.CATEGORICAL_COLUMNS`` (confirmado
        por el EDA de la compañera: ``["SEX", "EDUCATION", "MARRIAGE"]``).
    apply_scaling : bool, default=False

    Returns
    -------
    sklearn.compose.ColumnTransformer
    """
    cat_cols = categorical_columns if categorical_columns is not None else list(config.CATEGORICAL_COLUMNS)
    numeric_transformer = StandardScaler() if apply_scaling else "passthrough"

    transformers = []
    if cat_cols:
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols)
        )

    return ColumnTransformer(transformers=transformers, remainder=numeric_transformer)


def get_preprocessing_steps(
    spec: ModelSpec, categorical_columns: list[str] | None = None
) -> list[tuple[str, ColumnTransformer]]:
    """Construye el ``preprocessing_steps`` listo para
    ``balancing.build_pipeline(..., preprocessing_steps=get_preprocessing_steps(spec))``.

    El scaling se activa/desactiva automáticamente según
    ``spec.requires_scaling`` — quien orquesta el experimento (bloques 5/6)
    no necesita decidirlo modelo por modelo.

    Parameters
    ----------
    spec : ModelSpec
        Modelo base (ver ``models.MODEL_REGISTRY``).
    categorical_columns : list[str], optional
        Ver ``build_column_transformer``.

    Returns
    -------
    list[tuple[str, ColumnTransformer]]
        Lista de un solo paso, con el nombre ``"preprocessing"``.
    """
    transformer = build_column_transformer(
        categorical_columns=categorical_columns, apply_scaling=spec.requires_scaling
    )
    return [("preprocessing", transformer)]


if __name__ == "__main__":
    import json

    print(json.dumps(save_final_dataset(), indent=2, ensure_ascii=False))
    print(f"Guardado en {config.PROCESSED_DATA_PATH}")
