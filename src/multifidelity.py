"""Optimización multi-fidelidad: Successive Halving y Hyperband.

Los cuatro métodos del proyecto (grid, random, bayesiana, genética) evalúan
CADA configuración con la fidelidad completa: validación cruzada interna con
todas las filas del fold de entrenamiento externo. Un método multi-fidelidad
evalúa muchas configuraciones con una versión barata (menos filas) y solo
promueve a la fidelidad completa a las que van ganando.

Implementación propia (sin librerías de optimización), siguiendo Jamieson y
Talwalkar (2016) para Successive Halving y Li et al. (2018) para Hyperband:

- **Fidelidad** ``r`` en ``(0, 1]``: fracción de las filas de entrenamiento de
  cada fold interno (submuestra ESTRATIFICADA, semilla fija). El fold de
  validación interno se usa completo en cualquier fidelidad, así los scores
  de distintas fidelidades se miden sobre los mismos datos y solo cambia
  cuánto vio el modelo al entrenar. Todo (preprocesamiento, balanceo, modelo)
  se ajusta dentro del pipeline, sobre la submuestra: no hay fuga.
- **Successive Halving (SH)**: parte de ``n`` configuraciones muestreadas al
  azar con fidelidad ``r_0``; en cada ronda conserva el mejor ``1/eta`` y
  multiplica la fidelidad por ``eta``, hasta llegar a ``r = 1``.
- **Hyperband**: SH sufre del dilema «muchas configuraciones con poca
  fidelidad» vs. «pocas con mucha». Hyperband corre varias *brackets* de SH
  con distinto balance (``s = s_max ... 0``) y se queda con el mejor
  candidato evaluado a fidelidad completa.

El espacio de búsqueda es el mismo de ``random_search``
(``models.get_param_distributions``), así que la comparación con él es
directa: mismo espacio, misma métrica, misma validación interna.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import ParameterSampler, train_test_split

from src import config
from src.balancing import build_pipeline
from src.models import ModelSpec, get_param_distributions
from src.optimization import OptimizationResult, _n_splits


def _subsample_stratified(
    X: pd.DataFrame, y: pd.Series, fraction: float, random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Submuestra estratificada (conserva la proporción de clases)."""
    if fraction >= 1.0:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=fraction, stratify=y, random_state=random_state
    )
    return X_sub, y_sub


def _evaluate_config(
    params: dict[str, Any],
    fidelity: float,
    spec: ModelSpec,
    technique: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    scorer: Any,
    preprocessing_steps: list[tuple[str, Any]] | None,
    random_state: int,
) -> tuple[float, float]:
    """Score medio de validación interna con la fidelidad dada, y su costo (s)."""
    t_start = time.perf_counter()
    fold_scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, y_tr = _subsample_stratified(
            X.iloc[train_idx], y.iloc[train_idx], fidelity, random_state
        )
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        pipe = build_pipeline(
            spec,
            technique,
            y_train=y_tr,
            preprocessing_steps=preprocessing_steps,
            random_state=random_state,
            **params,
        )
        pipe.fit(X_tr, y_tr)
        fold_scores.append(scorer(pipe, X_val, y_val))
    return float(np.mean(fold_scores)), time.perf_counter() - t_start


def _successive_halving_bracket(
    configs: list[dict[str, Any]],
    min_fidelity: float,
    eta: int,
    log: list[dict[str, Any]],
    bracket: int,
    evaluate,
) -> None:
    """Una bracket de SH sobre ``configs``; agrega cada evaluación a ``log``."""
    n_rounds = int(round(math.log(1.0 / min_fidelity, eta))) + 1
    survivors = list(configs)
    for round_index in range(n_rounds):
        fidelity = min(1.0, min_fidelity * eta**round_index)
        scored = []
        for params in survivors:
            score, cost = evaluate(params, fidelity)
            log.append(
                {
                    "bracket": bracket,
                    "round": round_index,
                    "fidelity": fidelity,
                    "score": score,
                    "cost_seconds": cost,
                    "params": params,
                }
            )
            scored.append((score, params))
        scored.sort(key=lambda item: item[0], reverse=True)
        keep = max(1, int(len(scored) / eta))
        survivors = [params for _, params in scored[:keep]]


def run_hyperband(
    spec: ModelSpec,
    technique: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    scoring: str = config.DEFAULT_SCORING,
    eta: int = 3,
    min_fidelity: float = 1.0 / 9.0,
    n_iterations: int = 3,
    random_state: int = config.RANDOM_STATE,
    preprocessing_steps: list[tuple[str, Any]] | None = None,
) -> OptimizationResult:
    """Hyperband (Li et al., 2018) con fidelidad = fracción de filas.

    Parameters
    ----------
    eta : int, default=3
        Factor de reducción: cada ronda conserva ``1/eta`` de las
        configuraciones y multiplica la fidelidad por ``eta``.
    min_fidelity : float, default=1/9
        Fidelidad mínima (``eta ** -s_max``). Con ``eta=3`` hay 3 brackets
        (``s = 2, 1, 0``).
    n_iterations : int, default=3
        Veces que se repite el ciclo completo de brackets. Con los valores
        por defecto el costo TEÓRICO (si el costo creciera linealmente con la
        fidelidad) es de unas 26 evaluaciones a fidelidad completa, comparable
        al de ``random_search`` (25 configuraciones). En la práctica el costo
        de una evaluación con 1/9 de las filas es más de 1/9 del completo
        (hay costos fijos de preprocesamiento y de ajuste), así que el costo
        real se mide y se reporta (``full_fidelity_equivalents`` y
        ``cum_cost_seconds``) en vez de suponerlo.

    Returns
    -------
    OptimizationResult
        ``best_params`` es la mejor configuración entre las evaluadas a
        FIDELIDAD COMPLETA (los scores de fidelidad baja no son comparables
        con los de fidelidad completa y no compiten por el título).
        ``diagnostics["trace"]`` trae una fila por evaluación (cualquier
        fidelidad) con ``fidelity``, ``cost_seconds`` y ``best_so_far``
        (mejor score a fidelidad completa visto hasta ese momento; ``NaN``
        mientras no haya ninguno). ``diagnostics["full_fidelity_equivalents"]``
        es el costo total medido en evaluaciones completas.
    """
    scorer = get_scorer(scoring)
    s_max = int(round(math.log(1.0 / min_fidelity, eta)))
    distributions = get_param_distributions(spec)

    def evaluate(params: dict[str, Any], fidelity: float) -> tuple[float, float]:
        return _evaluate_config(
            params, fidelity, spec, technique, X, y, cv, scorer, preprocessing_steps, random_state
        )

    log: list[dict[str, Any]] = []
    sampler_seed = random_state
    for _iteration in range(n_iterations):
        for s in range(s_max, -1, -1):
            n_configs = int(math.ceil((s_max + 1) / (s + 1) * eta**s))
            bracket_min_fidelity = eta ** (-s)
            configs = list(ParameterSampler(distributions, n_iter=n_configs, random_state=sampler_seed))
            sampler_seed += 1
            _successive_halving_bracket(
                configs, bracket_min_fidelity, eta, log, bracket=s, evaluate=evaluate
            )

    trace = pd.DataFrame(log)
    trace.insert(0, "eval", np.arange(1, len(trace) + 1))
    trace["cum_cost_seconds"] = trace["cost_seconds"].cumsum()
    full_scores = trace["score"].where(trace["fidelity"] >= 1.0)
    trace["best_so_far"] = full_scores.cummax()

    full = trace[trace["fidelity"] >= 1.0]
    best_row = full.loc[full["score"].idxmax()]
    best_params = dict(best_row["params"])

    best_pipeline = build_pipeline(
        spec,
        technique,
        y_train=y,
        cv_n_splits=_n_splits(cv),
        preprocessing_steps=preprocessing_steps,
        random_state=random_state,
        **best_params,
    )
    best_pipeline.fit(X, y)

    full_cost = full["cost_seconds"].mean()
    return OptimizationResult(
        method="hyperband",
        best_params=best_params,
        best_score=float(best_row["score"]),
        best_pipeline=best_pipeline,
        n_evaluations=len(trace),
        diagnostics={
            "eta": eta,
            "min_fidelity": min_fidelity,
            "n_iterations": n_iterations,
            "trace": trace.drop(columns=["params"]),
            "full_fidelity_equivalents": float(trace["cost_seconds"].sum() / full_cost),
        },
    )
