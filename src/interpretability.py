"""Interpretabilidad del mejor modelo identificado: SHAP (global y local) y
comparación con LIME.

- **SHAP global**: summary plot sobre una muestra del fold de prueba.
- **SHAP local**: waterfall plot para 2 observaciones — una bien
  clasificada (alta confianza, correcta) y una con error alto (alta
  confianza, INCORRECTA) — ver ``select_representative_instances``.
- **LIME vs. SHAP** (específicamente para XGBoost, por pedido del curso):
  compara los top-k features de ambos métodos para la MISMA observación y
  discute cuándo y por qué pueden divergir (ver docstring de
  ``compare_shap_lime``).

**Sobre qué espacio de features se explica:** SHAP/LIME se calculan sobre
el espacio TRANSFORMADO (post ``preprocessing.py`` — one-hot de
``SEX``/``EDUCATION``/``MARRIAGE`` + scaling condicional), porque es el
espacio real que ve el modelo. ``aggregate_onehot_shap`` suma de vuelta
las contribuciones de las columnas one-hot a su variable categórica
original, para que el summary plot no fragmente ``EDUCATION`` en 4 barras
separadas.

**Costo computacional:** para modelos NO basados en árboles (KNN, Naive
Bayes, Regresión Logística, SVM), SHAP recurre a un explicador
model-agnostic (permutación/Kernel SHAP), mucho más lento que
``TreeExplainer``. Por eso ``background``/``n_explain`` se limitan
explícitamente por defecto — este módulo está pensado para explicar UN
modelo (el mejor identificado), no las 112 combinaciones.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.balancing import build_pipeline
from src.complexity_analysis import recover_outer_fold
from src.models import ModelSpec
from src.preprocessing import get_preprocessing_steps

TREE_MODELS: tuple[str, ...] = ("decision_tree", "random_forest", "xgboost")

# Para explicadores model-agnostic (no basados en árboles) — ver docstring del módulo.
DEFAULT_BACKGROUND_SIZE = 50
DEFAULT_N_EXPLAIN = 100

DISCUSSION_LIME_VS_SHAP = """
LIME (Local Interpretable Model-agnostic Explanations) ajusta un modelo
SUSTITUTO lineal (regularizado, ej. Lasso) sobre muestras perturbadas
alrededor de la observación a explicar, ponderadas por cercanía (kernel de
similitud) — es una aproximación LOCAL, válida solo en esa vecindad, y su
resultado depende de decisiones de diseño (ancho del kernel, número de
muestras perturbadas, número de features retenidas).

SHAP calcula valores de Shapley: el aporte marginal promedio de cada
feature sobre TODAS las posibles coaliciones de features, un concepto de
teoría de juegos cooperativos. Para modelos de árboles, TreeExplainer los
calcula de forma EXACTA y eficiente (no aproximada). SHAP satisface
axiomas de consistencia y precisión local (la suma de los valores SHAP más
el valor base reconstruye exactamente la predicción del modelo) que LIME
no garantiza.

**Cuándo divergen:**
1. **Features correlacionadas**: SHAP reparte el crédito entre features
   correlacionadas según su contribución marginal promedio; LIME, al
   ajustar un modelo lineal regularizado (Lasso), puede asignar todo el
   peso a una sola de las features correlacionadas y anular la otra —
   relevante aquí para el bloque `BILL_AMT1`-`BILL_AMT6` (VIF alto, ver
   EDA de la compañera).
2. **No linealidad local fuerte**: si la frontera de decisión es muy
   no lineal en la vecindad de la observación (ej. el efecto no lineal de
   `PAY_0` que el EDA detectó en SHAP dependence plots), un sustituto
   LINEAL como el de LIME la aproxima pobremente, mientras que SHAP
   (exacto para árboles) sigue reflejando la no linealidad real del modelo.
3. **Sensibilidad a hiperparámetros de LIME**: el ancho del kernel y el
   número de muestras perturbadas de LIME son elegidos por el usuario y
   pueden cambiar el ranking de importancia entre corridas — SHAP
   (TreeExplainer) es determinístico dado el modelo.
""".strip()


def select_representative_instances(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, int]:
    """Elige, DENTRO de un fold externo, la posición (índice posicional,
    usable con ``X_test.iloc[[pos]]``) de dos observaciones:

    - ``"well_classified"``: correcta, con la MENOR discrepancia
      |y_true - y_proba| (la predicción más confiadamente correcta).
    - ``"high_error"``: incorrecta, con la MAYOR discrepancia
      |y_true - y_proba| (la predicción más confiadamente equivocada).

    Acepta arrays crudos (no un ``OuterFoldResult``) para poder usarse
    tanto con un resultado en memoria (``fold_result.y_true`` etc.) como
    con predicciones recién generadas en una sesión nueva (ej.
    ``fit_explainable_pipeline`` + ``pipeline.predict``/``predict_proba``
    en ``06_interpretability.ipynb``, que no tiene un ``OuterFoldResult``
    a mano).

    Parameters
    ----------
    y_true, y_pred, y_proba : np.ndarray
        Del fold de prueba EXTERNO (ver ``nested_cv.OuterFoldResult`` o
        ``fit_explainable_pipeline``).

    Returns
    -------
    dict[str, int]
    """
    y_true, y_pred, y_proba = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_proba)
    correct = y_true == y_pred
    error_magnitude = np.abs(y_true - y_proba)

    correct_idx = np.where(correct)[0]
    incorrect_idx = np.where(~correct)[0]
    if len(correct_idx) == 0 or len(incorrect_idx) == 0:
        raise ValueError(
            "Este fold no tiene simultáneamente ejemplos correctos e incorrectos; "
            "elegir otro fold_index (ver complexity_analysis.select_best_fold_index)."
        )

    return {
        "well_classified": int(correct_idx[np.argmin(error_magnitude[correct_idx])]),
        "high_error": int(incorrect_idx[np.argmax(error_magnitude[incorrect_idx])]),
    }


def fit_explainable_pipeline(
    spec: ModelSpec,
    technique: str,
    best_params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    fold_index: int,
    n_outer_folds: int = config.N_OUTER_FOLDS,
    random_state: int = config.RANDOM_STATE,
    categorical_columns: list[str] | None = None,
):
    """Reconstruye el fold, ajusta el pipeline (technique + best_params) y
    devuelve todo lo necesario para explicarlo: el pipeline ajustado, los
    datos crudos y transformados, y los nombres de las features
    transformadas (post one-hot/scaling).

    Returns
    -------
    dict con ``pipeline`` (ajustado), ``X_train``, ``X_test``, ``y_train``,
    ``y_test`` (crudos), ``X_train_t``, ``X_test_t`` (transformados, listos
    para el modelo/SHAP), ``feature_names`` (post-transformación).
    """
    X_train, X_test, y_train, y_test = recover_outer_fold(
        X, y, fold_index, n_outer_folds=n_outer_folds, random_state=random_state
    )
    preprocessing_steps = get_preprocessing_steps(spec, categorical_columns=categorical_columns)
    pipeline = build_pipeline(
        spec,
        technique,
        y_train=y_train,
        preprocessing_steps=preprocessing_steps,
        random_state=random_state,
        **best_params,
    )
    pipeline.fit(X_train, y_train)

    preprocessor = pipeline.named_steps["preprocessing"]
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out())

    return {
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_t": X_train_t,
        "X_test_t": X_test_t,
        "feature_names": feature_names,
    }


def _positive_class_values(raw_values: np.ndarray) -> np.ndarray:
    """Normaliza la salida de distintos explicadores SHAP a un array 2D
    (n_samples, n_features) de la clase POSITIVA.

    Distintas combinaciones de versión de ``shap``/tipo de modelo devuelven
    a veces (n_samples, n_features) directamente y a veces
    (n_samples, n_features, n_clases) — este helper unifica ambos casos
    para que el resto del módulo no dependa de esos detalles de versión.
    """
    values = np.asarray(raw_values)
    if values.ndim == 3:
        return values[:, :, -1]
    return values


def compute_shap_values(
    model_name: str,
    fitted_model: Any,
    X_background,
    X_explain,
    background_size: int = DEFAULT_BACKGROUND_SIZE,
    random_state: int = config.RANDOM_STATE,
):
    """Calcula valores SHAP para ``X_explain`` (clase positiva).

    Usa ``shap.TreeExplainer`` (exacto, rápido) para modelos de árboles
    (``TREE_MODELS``); para el resto, ``shap.Explainer`` model-agnostic
    sobre ``predict_proba`` con un background MUESTREADO (ver constante
    ``DEFAULT_BACKGROUND_SIZE`` — Kernel/Permutation SHAP escalan mal con
    el tamaño del background).

    Returns
    -------
    tuple(np.ndarray de forma (n_explain, n_features), base_value: float)
    """
    import shap

    if model_name in TREE_MODELS:
        explainer = shap.TreeExplainer(fitted_model)
        raw = explainer(X_explain)
    else:
        rng = np.random.default_rng(random_state)
        n_bg = min(background_size, X_background.shape[0])
        bg_idx = rng.choice(X_background.shape[0], size=n_bg, replace=False)
        background = X_background[bg_idx] if not hasattr(X_background, "iloc") else X_background.iloc[bg_idx]
        explainer = shap.Explainer(fitted_model.predict_proba, background)
        raw = explainer(X_explain)

    values = _positive_class_values(raw.values)
    base_value = raw.base_values
    base_value = float(np.ravel(base_value)[-1]) if np.ndim(base_value) > 0 else float(base_value)
    return values, base_value


def aggregate_onehot_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
    categorical_columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Suma las contribuciones SHAP de las columnas one-hot de vuelta a su
    variable categórica original (ej. ``EDUCATION_1``, ``EDUCATION_2``, ...
    -> ``EDUCATION``), para que el summary plot muestre UNA barra por
    variable categórica original en vez de fragmentarla.

    ``sklearn.compose.ColumnTransformer.get_feature_names_out()`` nombra
    las columnas one-hot como ``"categorical__<col>_<categoria>"`` — se
    detecta el prefijo ``"categorical__<col>_"`` para cada nombre en
    ``config.CATEGORICAL_COLUMNS`` (o ``categorical_columns``) y se agrupan.
    """
    categorical_columns = categorical_columns if categorical_columns is not None else list(config.CATEGORICAL_COLUMNS)
    grouped_indices, remaining_idx = _group_onehot_columns(feature_names, categorical_columns)

    grouped_names = list(grouped_indices.keys())
    remaining_names = [feature_names[i].replace("remainder__", "") for i in remaining_idx]

    all_names = grouped_names + remaining_names
    grouped_arrays = [shap_values[:, idx].sum(axis=1) for idx in grouped_indices.values()]
    all_values = np.column_stack(grouped_arrays + [shap_values[:, remaining_idx]])
    return all_values, all_names


def _group_onehot_columns(
    feature_names: list[str], categorical_columns: list[str]
) -> tuple[dict[str, list[int]], list[int]]:
    """Agrupa los índices de columnas one-hot (``"categorical__<col>_<cat>"``)
    por su variable categórica original. Compartido por
    ``aggregate_onehot_shap`` y ``build_aggregated_display_matrix`` para no
    duplicar la lógica de detección de prefijos.
    """
    grouped_indices: dict[str, list[int]] = {}
    consumed: set[int] = set()
    for cat_col in categorical_columns:
        prefix = f"categorical__{cat_col}_"
        matching_idx = [i for i, name in enumerate(feature_names) if name.startswith(prefix)]
        if not matching_idx:
            continue
        grouped_indices[cat_col] = matching_idx
        consumed.update(matching_idx)
    remaining_idx = [i for i in range(len(feature_names)) if i not in consumed]
    return grouped_indices, remaining_idx


def build_aggregated_display_matrix(
    X_raw: pd.DataFrame,
    X_transformed: np.ndarray,
    feature_names: list[str],
    categorical_columns: list[str] | None = None,
) -> np.ndarray:
    """Construye una matriz de "valores de feature" alineada columna-por-
    columna con la salida de ``aggregate_onehot_shap`` — para colorear el
    summary plot por el valor real de cada feature.

    Para las columnas categóricas agregadas, usa el código crudo ORIGINAL
    (de ``X_raw``, ej. ``EDUCATION`` en 1..4) en vez de sumar los
    indicadores one-hot (que no tendría un significado interpretable).
    Para el resto, usa el valor ya transformado (post scaling condicional).

    Parameters
    ----------
    X_raw : pd.DataFrame
        Mismas filas, mismo orden, que ``X_transformed`` (ej.
        ``fitted_explainable_pipeline["X_test"]``).
    X_transformed : np.ndarray
        Salida de ``preprocessor.transform(X_raw)``.
    feature_names : list[str]
        Nombres POST-transformación (antes de agregar), ver
        ``fit_explainable_pipeline``.
    """
    categorical_columns = categorical_columns if categorical_columns is not None else list(config.CATEGORICAL_COLUMNS)
    grouped_indices, remaining_idx = _group_onehot_columns(feature_names, categorical_columns)

    grouped_arrays = [X_raw[cat_col].to_numpy() for cat_col in grouped_indices]
    return np.column_stack(grouped_arrays + [X_transformed[:, remaining_idx]])


def plot_shap_summary(
    shap_values: np.ndarray,
    X_display: np.ndarray,
    feature_names: list[str],
    save_path: str | None = None,
    max_display: int = 15,
) -> None:
    """Summary plot global (importancia + magnitud) — guarda a PNG si se
    provee ``save_path`` (ej. ``results/figures/shap_summary_<modelo>.png``)."""
    import matplotlib.pyplot as plt
    import shap

    shap.summary_plot(
        shap_values, X_display, feature_names=feature_names, max_display=max_display, show=False
    )
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_shap_waterfall(
    shap_values_row: np.ndarray,
    base_value: float,
    X_row: np.ndarray,
    feature_names: list[str],
    save_path: str | None = None,
    max_display: int = 12,
) -> None:
    """Waterfall plot LOCAL para una sola observación."""
    import matplotlib.pyplot as plt
    import shap

    explanation = shap.Explanation(
        values=shap_values_row,
        base_values=base_value,
        data=X_row,
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# LIME (específicamente para XGBoost, ver docstring del módulo)
# ---------------------------------------------------------------------------


def explain_with_lime(
    fitted_model: Any,
    X_train_t: np.ndarray,
    feature_names: list[str],
    x_explain: np.ndarray,
    class_names: tuple[str, str] = ("no_default", "default"),
    num_features: int = 10,
    random_state: int = config.RANDOM_STATE,
):
    """Explica UNA observación con LIME, sobre el mismo espacio
    transformado que ve el modelo (post preprocessing).

    Returns
    -------
    lime.explanation.Explanation
    """
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=np.asarray(X_train_t),
        feature_names=feature_names,
        class_names=list(class_names),
        mode="classification",
        random_state=random_state,
    )
    return explainer.explain_instance(
        np.asarray(x_explain).ravel(),
        fitted_model.predict_proba,
        num_features=num_features,
    )


def compare_shap_lime(
    shap_values_row: np.ndarray,
    lime_explanation: Any,
    feature_names: list[str],
    top_k: int = 10,
) -> pd.DataFrame:
    """Compara, para la MISMA observación, el top-``top_k`` de features
    según |SHAP| vs. el ranking/coeficientes de LIME.

    Ver ``DISCUSSION_LIME_VS_SHAP`` (constante de este módulo) para la
    discusión metodológica completa de por qué pueden divergir — esta
    función da la evidencia NUMÉRICA concreta para esa discusión en el
    reporte, no la reemplaza.

    Returns
    -------
    pd.DataFrame con columnas ``feature``, ``shap_value``, ``lime_weight``
    (``NaN`` si la feature no está en el top-``num_features`` de LIME),
    ordenado por |shap_value| descendente.
    """
    shap_rank = np.argsort(-np.abs(shap_values_row))[:top_k]
    lime_weights = dict(lime_explanation.as_list())

    rows = []
    for idx in shap_rank:
        name = feature_names[idx]
        lime_weight = next((w for feat_desc, w in lime_weights.items() if name in feat_desc), np.nan)
        rows.append({"feature": name, "shap_value": float(shap_values_row[idx]), "lime_weight": lime_weight})
    return pd.DataFrame(rows).sort_values("shap_value", key=np.abs, ascending=False).reset_index(drop=True)
