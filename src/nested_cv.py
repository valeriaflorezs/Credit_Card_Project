"""Validación cruzada anidada: bucle externo (outer loop) para estimar
desempeño de generalización, bucle interno (inner loop) para optimizar
hiperparámetros — nunca al revés.

**Regla crítica del curso:** la métrica final reportada SIEMPRE proviene
de evaluar el pipeline ya optimizado sobre el fold de prueba del OUTER
loop; el ``best_score`` que devuelve ``optimization.py`` es solo la
métrica de validación INTERNA (inner loop) usada para elegir
hiperparámetros, y nunca debe confundirse con la métrica final —
mezclar ambos niveles es, según el curso, la causa más común de
sobreestimación de desempeño.

Estructura, por cada combinación (modelo, técnica de balanceo, método de
optimización):

1. ``outer_cv`` (``config.N_OUTER_FOLDS`` folds) parte ``(X, y)`` en
   fold de entrenamiento/prueba externo.
2. Dentro de cada fold externo, ``optimization.run_optimization`` corre
   el método de optimización elegido usando ``inner_cv``
   (``config.N_INNER_FOLDS`` folds) ÚNICAMENTE sobre el fold de
   entrenamiento externo, y devuelve un pipeline ya reajustado
   (``best_pipeline``) sobre ese fold de entrenamiento externo completo.
3. Ese ``best_pipeline`` se evalúa sobre el fold de PRUEBA externo (nunca
   visto por la optimización) — esas predicciones son las que alimentan
   las métricas finales.

Este módulo calcula un set "núcleo" de métricas (accuracy, precision,
recall, F1, ROC-AUC) por fold externo y su resumen media±desviación
estándar. El set completo de evaluación (matriz de confusión, curva ROC,
calibración/Brier/ECE) se construye en ``evaluation.py`` (bloque 8) a
partir de ``y_true``/``y_pred``/``y_proba`` que este módulo conserva por
fold — sin recalcular ni volver a ajustar nada.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src import config
from src.balancing import is_compatible
from src.models import ModelSpec
from src.optimization import OptimizationResult, run_optimization
from src.preprocessing import get_preprocessing_steps

CORE_METRICS: tuple[str, ...] = ("accuracy", "precision", "recall", "f1", "roc_auc")


@dataclass
class OuterFoldResult:
    """Resultado de un solo fold del outer loop.

    Attributes
    ----------
    fold_index : int
    best_params : dict
        Hiperparámetros elegidos por el inner loop para ESTE fold externo
        (pueden variar de un fold externo a otro — es normal y esperado).
    inner_best_score : float
        Score de validación INTERNA del inner loop. Solo referencia/
        diagnóstico (ej. detectar overfitting del inner loop si difiere
        mucho de las métricas de este mismo fold) — NUNCA se reporta como
        métrica final.
    y_true, y_pred, y_proba : np.ndarray
        Del fold de prueba EXTERNO. ``y_proba`` es la probabilidad de la
        clase positiva. Conservados para ``evaluation.py`` (bloque 8):
        matriz de confusión, curva ROC, calibración.
    metrics : dict[str, float]
        Accuracy/precision/recall/F1/ROC-AUC de ESTE fold externo.
    fit_time_seconds : float
        Tiempo total del proceso de optimización + refit en este fold.
    n_evaluations : int
        Configuraciones de hiperparámetros evaluadas por el optimizador
        en este fold (costo computacional).
    """

    fold_index: int
    best_params: dict[str, Any]
    inner_best_score: float
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    metrics: dict[str, float]
    fit_time_seconds: float
    n_evaluations: int


@dataclass
class NestedCVResult:
    """Resultado completo de validación cruzada anidada para UNA
    combinación (modelo, técnica de balanceo, método de optimización).

    Attributes
    ----------
    model_name, technique, method : str
    fold_results : list[OuterFoldResult]
        Un elemento por fold del outer loop.
    metrics_summary : dict[str, tuple[float, float]]
        ``{métrica: (media, desviación_estándar)}`` a través de los folds
        del OUTER loop — esta es la fuente de verdad para reportar
        desempeño (ver regla crítica en el docstring del módulo).
    total_time_seconds : float
    """

    model_name: str
    technique: str
    method: str
    fold_results: list[OuterFoldResult] = field(default_factory=list)
    metrics_summary: dict[str, tuple[float, float]] = field(default_factory=dict)
    total_time_seconds: float = 0.0

    def summary_row(self) -> dict[str, Any]:
        """Una fila lista para la tabla maestra de experimentos (bloque 6):
        ``{modelo, tecnica, metodo, <metrica>_mean, <metrica>_std, ...}``.

        También incluye ``<metrica>_per_fold`` (lista JSON, un valor por
        fold externo) y ``best_params_per_fold`` (lista JSON de dicts).
        Sin esto, una tabla maestra cargada en una sesión NUEVA (ej.
        ``05_results_analysis.ipynb``, corrido después de
        ``04_run_experiments.ipynb``) solo tendría medias/desviaciones —
        insuficiente para Friedman/Nemenyi (bloque 10, necesitan el valor
        POR fold, no el resumen) o para recalcular matriz de confusión/ROC/
        calibración/SHAP de una combinación puntual sin repetir la
        búsqueda de hiperparámetros (basta con los ``best_params`` ya
        encontrados, ver ``evaluation.regenerate_outof_fold_predictions``).
        """
        import json

        row: dict[str, Any] = {
            "model": self.model_name,
            "technique": self.technique,
            "method": self.method,
            "total_time_seconds": self.total_time_seconds,
            "n_outer_folds": len(self.fold_results),
        }
        for metric_name, (mean, std) in self.metrics_summary.items():
            row[f"{metric_name}_mean"] = mean
            row[f"{metric_name}_std"] = std
            row[f"{metric_name}_per_fold"] = json.dumps(
                [fr.metrics[metric_name] for fr in self.fold_results]
            )
        row["best_params_per_fold"] = json.dumps([fr.best_params for fr in self.fold_results])
        return row


def compute_core_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Accuracy, precision, recall, F1 y ROC-AUC de un solo fold externo.

    ``zero_division=0`` evita que un fold sin predicciones positivas
    (posible con folds pequeños y clases muy desbalanceadas) haga fallar
    todo el experimento — un valor 0 en ese caso es la convención estándar
    de scikit-learn y queda visible en el resultado, no oculto.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def _summarize_metrics(fold_results: list[OuterFoldResult]) -> dict[str, tuple[float, float]]:
    summary: dict[str, tuple[float, float]] = {}
    for metric_name in CORE_METRICS:
        values = [fr.metrics[metric_name] for fr in fold_results]
        summary[metric_name] = (float(np.mean(values)), float(np.std(values)))
    return summary


def run_nested_cv(
    spec: ModelSpec,
    technique: str,
    method: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_outer_folds: int = config.N_OUTER_FOLDS,
    n_inner_folds: int = config.N_INNER_FOLDS,
    scoring: str = config.DEFAULT_SCORING,
    random_state: int = config.RANDOM_STATE,
    categorical_columns: list[str] | None = None,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> NestedCVResult:
    """Corre validación cruzada anidada completa para una combinación
    (modelo, técnica de balanceo, método de optimización).

    Parameters
    ----------
    spec : ModelSpec
        Modelo base (ver ``models.MODEL_REGISTRY``).
    technique : str
        Técnica de balanceo (ver ``balancing.BALANCING_TECHNIQUES``). Debe
        ser compatible con ``spec`` (ver ``balancing.is_compatible``) — de
        lo contrario se lanza ``ValueError`` antes de correr ningún fold.
    method : str
        Método de optimización (ver ``optimization.OPTIMIZATION_METHODS``).
    X, y : DataFrame, Series
        Dataset COMPLETO (ya deduplicado si corresponde — ver
        ``preprocessing.remove_duplicate_rows``, que debe aplicarse antes,
        una sola vez, nunca dentro de este bucle).
    n_outer_folds, n_inner_folds : int
        Folds del outer/inner loop (defaults: ``config.N_OUTER_FOLDS``/
        ``config.N_INNER_FOLDS``).
    scoring : str, default=config.DEFAULT_SCORING
        Métrica que optimiza el inner loop (ver ``sklearn.metrics.get_scorer``).
        Nota: el inner loop puede optimizar una métrica distinta a las que
        se reportan en ``metrics_summary`` (que siempre incluye las 5
        ``CORE_METRICS``); ver ``evaluation.py`` para la justificación de
        cuál debería ser la métrica principal dado el desbalance del dataset.
    random_state : int, default=config.RANDOM_STATE
    categorical_columns : list[str], optional
        Ver ``preprocessing.get_preprocessing_steps``. Si no se provee,
        usa ``config.CATEGORICAL_COLUMNS``.
    optimizer_kwargs : dict, optional
        Argumentos extra para el método de optimización elegido (ej.
        ``{"n_iter": 30}`` para random_search, ``{"population_size": 20,
        "n_generations": 15}`` para genetic_deap).

    Returns
    -------
    NestedCVResult
    """
    if not is_compatible(spec, technique):
        raise ValueError(
            f"La técnica de balanceo '{technique}' no es compatible con el modelo "
            f"'{spec.name}' (supports_class_weight={spec.supports_class_weight})."
        )

    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)
    preprocessing_steps = get_preprocessing_steps(spec, categorical_columns=categorical_columns)

    fold_results: list[OuterFoldResult] = []
    t_start_total = time.time()

    for fold_index, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]

        t0 = time.time()
        opt_result: OptimizationResult = run_optimization(
            method,
            spec,
            technique,
            X_train_outer,
            y_train_outer,
            inner_cv,
            scoring=scoring,
            random_state=random_state,
            preprocessing_steps=preprocessing_steps,
            **(optimizer_kwargs or {}),
        )
        fit_time_seconds = time.time() - t0

        # best_pipeline ya viene reajustado sobre TODO X_train_outer
        # (contrato de optimization.OptimizationResult) — se evalúa aquí
        # sobre X_test_outer, nunca visto durante la optimización.
        y_pred = opt_result.best_pipeline.predict(X_test_outer)
        y_proba = opt_result.best_pipeline.predict_proba(X_test_outer)[:, 1]
        y_true = y_test_outer.to_numpy()

        fold_results.append(
            OuterFoldResult(
                fold_index=fold_index,
                best_params=opt_result.best_params,
                inner_best_score=opt_result.best_score,
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                metrics=compute_core_metrics(y_true, y_pred, y_proba),
                fit_time_seconds=fit_time_seconds,
                n_evaluations=opt_result.n_evaluations,
            )
        )

    total_time_seconds = time.time() - t_start_total

    return NestedCVResult(
        model_name=spec.name,
        technique=technique,
        method=method,
        fold_results=fold_results,
        metrics_summary=_summarize_metrics(fold_results),
        total_time_seconds=total_time_seconds,
    )
