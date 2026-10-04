"""Wrappers de las 4 técnicas de balanceo de clases.

- ``"none"``: sin balanceo (línea base).
- ``"smote"``: oversampling sintético de la clase minoritaria.
- ``"adasyn"``: oversampling adaptativo (más síntesis donde la clasificación
  es más difícil).
- ``"class_weight"``: no resamplea los datos — pondera la función de
  pérdida del modelo (ver ``models.py::ModelSpec.supports_class_weight``).

**Regla crítica (evita data leakage):** SMOTE y ADASYN se envuelven en
``imblearn.pipeline.Pipeline`` (nunca ``sklearn.pipeline.Pipeline``), de
forma que el resampling se ajusta y aplica ÚNICAMENTE sobre el fold de
ENTRENAMIENTO en cada iteración de validación cruzada — ``imblearn``
garantiza que los pasos de resampling se omiten automáticamente durante
``.predict()``/``.predict_proba()``, por lo que el fold de
validación/test nunca se ve alterado.

**Nota sobre 'class_weight' en XGBoost:** a diferencia de scikit-learn
(donde ``class_weight='balanced'`` se recalcula automáticamente dentro de
cada llamada a ``fit()``, por lo que no hay riesgo de leakage ni siquiera
dentro de un ``GridSearchCV``), XGBoost no tiene ``class_weight`` — usa
``scale_pos_weight``, un hiperparámetro fijo que debe conocerse ANTES de
llamar a ``fit()``. Por eso ``build_pipeline`` recibe ``y_train`` y calcula
``scale_pos_weight`` una vez, a partir de esos datos de entrenamiento
(nunca del fold de validación/test — no hay leakage). La simplificación
consciente aquí es que ese valor queda fijo para todos los folds internos
que se deriven de ese ``y_train`` (ej. las particiones del inner loop
dentro de un mismo outer fold): recalcularlo por cada fold interno
requeriría un estimador envoltorio que complica la integración con
``GridSearchCV``/``RandomizedSearchCV``, y el ratio de clases es estable
bajo CV estratificada, así que la variación entre folds internos es
mínima. ``nested_cv.py`` debe invocar ``build_pipeline`` con el ``y_train``
del OUTER fold (antes de dividir en inner folds), nunca con el dataset
completo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import Memory
from sklearn.base import BaseEstimator

from src.config import JOBLIB_CACHE_DIR, RANDOM_STATE
from src.models import ModelSpec

BALANCING_TECHNIQUES: tuple[str, ...] = ("none", "smote", "adasyn", "class_weight")

# Cachea el resultado de los pasos de balanceo (SMOTE/ADASYN) en disco,
# indexado por el contenido de X/y de entrada. Sin esto, GridSearchCV/
# RandomizedSearchCV/Optuna/DEAP repiten el resampling una vez por cada
# combinación de hiperparámetros evaluada, aunque el resampling no dependa
# de esos hiperparámetros (solo del fold de entrenamiento y la técnica) —
# con 64 combinaciones de grid_search eso es 64x cómputo redundante de
# SMOTE/ADASYN. El cache es compartido entre modelos/métodos: dos
# combinaciones distintas que reciban el mismo fold de entrenamiento (ej.
# mismo outer fold, misma técnica) reutilizan el mismo resultado. No afecta
# a "none"/"class_weight" (no hay paso de resampling que cachear).
_PIPELINE_CACHE = Memory(location=str(JOBLIB_CACHE_DIR), verbose=0)


def is_compatible(spec: ModelSpec, technique: str) -> bool:
    """Indica si una técnica de balanceo aplica a un modelo dado.

    ``"class_weight"`` requiere que el modelo soporte ponderación de clases
    (ver ``ModelSpec.supports_class_weight`` — KNN y Naive Bayes no la
    soportan). Las demás técnicas (``none``, ``smote``, ``adasyn``) actúan
    sobre los datos de entrada y son compatibles con cualquier modelo.

    Parameters
    ----------
    spec : ModelSpec
    technique : {"none", "smote", "adasyn", "class_weight"}

    Returns
    -------
    bool
    """
    if technique not in BALANCING_TECHNIQUES:
        raise ValueError(f"Técnica de balanceo desconocida: '{technique}'")
    if technique == "class_weight":
        return spec.supports_class_weight
    return True


def _safe_k_neighbors(
    y: pd.Series | np.ndarray, default: int = 5, cv_n_splits: int | None = None
) -> int:
    """Ajusta ``k_neighbors``/``n_neighbors`` de SMOTE/ADASYN al tamaño real
    de la clase minoritaria del fold.

    SMOTE/ADASYN fallan si ``k_neighbors >= n_minority``. En validación
    cruzada anidada algunos folds de entrenamiento (sobre todo folds
    internos, más pequeños) pueden tener pocas muestras minoritarias, así
    que este valor se recalcula por fold en vez de fijarse globalmente.

    Parameters
    ----------
    cv_n_splits : int, optional
        Si se provee, asume que ``y`` representa el fold de entrenamiento
        EXTERNO (outer) y que este resampler se reutilizará dentro de una
        búsqueda (``GridSearchCV``/``RandomizedSearchCV``) que a su vez lo
        parte en ``cv_n_splits`` folds internos más pequeños. En ese caso
        se dimensiona ``k_neighbors`` con un margen conservador
        (``(cv_n_splits - 1) / cv_n_splits`` del conteo minoritario) para
        que siga siendo válido en el fold interno de entrenamiento más
        pequeño, no solo en el fold externo completo.
    """
    n_minority = int(pd.Series(y).value_counts().min())
    if cv_n_splits is not None and cv_n_splits > 1:
        n_minority = int(n_minority * (cv_n_splits - 1) / cv_n_splits)
    return max(1, min(default, n_minority - 1))


def get_resampler(
    technique: str,
    y_train: pd.Series | np.ndarray | None = None,
    random_state: int = RANDOM_STATE,
    cv_n_splits: int | None = None,
):
    """Construye el resampler de ``imblearn`` para ``"smote"``/``"adasyn"``.

    Parameters
    ----------
    technique : {"none", "smote", "adasyn", "class_weight"}
    y_train : array-like, optional
        Usado únicamente para ajustar ``k_neighbors``/``n_neighbors`` de
        forma segura (ver ``_safe_k_neighbors``). Si no se provee, se usa
        el default de la librería (5).
    random_state : int, default=config.RANDOM_STATE
    cv_n_splits : int, optional
        Ver ``_safe_k_neighbors``. Pasar el número de folds internos cuando
        este resampler se use dentro de ``GridSearchCV``/``RandomizedSearchCV``
        (optimization.py, bloque 4), para no fallar en el fold interno más chico.

    Returns
    -------
    imblearn resampler or None
        ``None`` para ``"none"``/``"class_weight"`` (no resamplean datos).
    """
    if technique in ("none", "class_weight"):
        return None
    if technique == "smote":
        k = _safe_k_neighbors(y_train, cv_n_splits=cv_n_splits) if y_train is not None else 5
        return SMOTE(random_state=random_state, k_neighbors=k)
    if technique == "adasyn":
        k = _safe_k_neighbors(y_train, cv_n_splits=cv_n_splits) if y_train is not None else 5
        return ADASYN(random_state=random_state, n_neighbors=k)
    raise ValueError(f"Técnica de balanceo desconocida: '{technique}'")


def compute_class_weight_kwargs(spec: ModelSpec, y_train: pd.Series | np.ndarray) -> dict[str, Any]:
    """Calcula el kwarg de balanceo por peso de clase para un modelo dado.

    La mayoría de modelos aceptan directamente ``class_weight='balanced'``,
    que scikit-learn recalcula automáticamente dentro de cada ``fit()`` —
    sin riesgo de leakage. XGBoost no tiene ``class_weight``: en su lugar
    se calcula ``scale_pos_weight = n_negativos / n_positivos`` sobre
    ``y_train`` (ver limitación documentada en el docstring del módulo).

    Parameters
    ----------
    spec : ModelSpec
    y_train : array-like
        Target del fold de ENTRENAMIENTO únicamente (nunca de
        validación/test ni del dataset completo).

    Returns
    -------
    dict
        Vacío si el modelo no soporta esta técnica (ver ``is_compatible``).
    """
    if not spec.supports_class_weight:
        return {}
    if spec.class_weight_param == "scale_pos_weight":
        counts = pd.Series(np.asarray(y_train)).value_counts()
        n_neg, n_pos = int(counts.get(0, 0)), int(counts.get(1, 0))
        if n_pos == 0:
            raise ValueError(
                "y_train no tiene ejemplos de la clase positiva; no se puede calcular scale_pos_weight."
            )
        return {spec.class_weight_param: n_neg / n_pos}
    return {spec.class_weight_param: "balanced"}


def build_pipeline(
    spec: ModelSpec,
    technique: str,
    y_train: pd.Series | np.ndarray | None = None,
    preprocessing_steps: list[tuple[str, BaseEstimator]] | None = None,
    random_state: int = RANDOM_STATE,
    cv_n_splits: int | None = None,
    **model_kwargs: Any,
) -> ImbPipeline:
    """Construye el pipeline completo (preprocesamiento + balanceo + modelo).

    Usa siempre ``imblearn.pipeline.Pipeline`` (nunca
    ``sklearn.pipeline.Pipeline``): si ``technique`` es ``"smote"`` o
    ``"adasyn"``, el resampling se ajusta y aplica ÚNICAMENTE sobre los
    datos que se le pasen a ``.fit()`` en cada iteración de validación
    cruzada — nunca sobre el fold de validación/test evaluado con
    ``.predict()``/``.predict_proba()``.

    Parameters
    ----------
    spec : ModelSpec
        Especificación del modelo base (ver ``models.py``).
    technique : {"none", "smote", "adasyn", "class_weight"}
        Técnica de balanceo a aplicar. Debe ser compatible con ``spec``
        (ver ``is_compatible``) — de lo contrario se lanza ``ValueError``.
    y_train : array-like, optional
        Fold de entrenamiento actual. Requerido para ``"class_weight"``
        (para calcular el peso) y recomendado para ``"smote"``/``"adasyn"``
        (para ajustar ``k_neighbors`` de forma segura).
    preprocessing_steps : list of (str, estimator), optional
        Pasos adicionales (ej. ``StandardScaler``) antes del balanceo/modelo,
        provistos por ``preprocessing.py``.
    random_state : int, default=config.RANDOM_STATE
    cv_n_splits : int, optional
        Pasar cuando este pipeline se reutilizará dentro de una búsqueda
        (``GridSearchCV``/``RandomizedSearchCV``, ver ``optimization.py``)
        que particiona ``y_train`` en folds internos más pequeños — ajusta
        ``k_neighbors`` de SMOTE/ADASYN de forma conservadora (ver
        ``get_resampler``/``_safe_k_neighbors``).
    **model_kwargs
        Hiperparámetros extra para la fábrica del modelo (ej. valores
        muestreados por un optimizador).

    Returns
    -------
    imblearn.pipeline.Pipeline
    """
    if not is_compatible(spec, technique):
        raise ValueError(
            f"La técnica de balanceo '{technique}' no es compatible con el modelo "
            f"'{spec.name}' (supports_class_weight={spec.supports_class_weight}). "
            "Verificar con is_compatible() antes de construir el pipeline."
        )

    steps: list[tuple[str, Any]] = list(preprocessing_steps or [])

    resampler = get_resampler(technique, y_train=y_train, random_state=random_state, cv_n_splits=cv_n_splits)
    if resampler is not None:
        steps.append((technique, resampler))

    extra_kwargs = dict(model_kwargs)
    if technique == "class_weight":
        if y_train is None:
            raise ValueError(
                "technique='class_weight' requiere y_train para calcular el peso "
                "(ej. scale_pos_weight en XGBoost, o 'balanced' para los demás modelos)."
            )
        extra_kwargs.update(compute_class_weight_kwargs(spec, y_train))

    estimator = spec.estimator_factory(random_state=random_state, **extra_kwargs)
    steps.append(("model", estimator))

    return ImbPipeline(steps=steps, memory=_PIPELINE_CACHE)


def get_compatible_techniques(spec: ModelSpec) -> list[str]:
    """Devuelve las técnicas de balanceo compatibles con un modelo, en orden fijo."""
    return [t for t in BALANCING_TECHNIQUES if is_compatible(spec, t)]
