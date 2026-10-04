"""Análisis de complejidad computacional: complejidad asintótica O(·)
teórica (train e inferencia) de cada modelo base, contrastada con tiempo y
memoria medidos empíricamente, y tabla comparativa modelo estándar
(hiperparámetros por defecto, sin balanceo) vs. modelo optimizado
(hiperparámetros del inner loop + técnica de balanceo elegida).

La medición de memoria usa ``tracemalloc`` (librería estándar, funciona en
Windows) en vez de ``resource.getrusage`` (solo POSIX, no disponible en el
hardware Windows de este proyecto) o ``psutil`` (dependencia extra
evitable). Mide el pico de memoria de las asignaciones de Python/NumPy
durante ``fit``/``predict`` — una aproximación razonable al costo de
memoria del modelo, no el RSS completo del proceso.
"""

from __future__ import annotations

import time
import tracemalloc
from itertools import islice
from typing import Any

import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold

from src import config
from src.balancing import build_pipeline
from src.models import MODEL_REGISTRY, ModelSpec
from src.nested_cv import NestedCVResult
from src.preprocessing import get_preprocessing_steps

# ---------------------------------------------------------------------------
# Complejidad teórica (metadata) — n = observaciones, p = features, T =
# número de árboles/rondas de boosting, k = n_neighbors, s = vectores de
# soporte.
# ---------------------------------------------------------------------------

TRAIN_INFERENCE_COMPLEXITY: dict[str, dict[str, str]] = {
    "knn": {
        "train": "O(n·p) — solo almacena los datos (\"lazy learner\"), sin ajuste real.",
        "inference": (
            "O(n·p) por consulta con búsqueda exhaustiva (fuerza bruta). "
            "KD-Tree/Ball-Tree: O(p·log n) por consulta en dimensión baja/moderada "
            "(degradan hacia fuerza bruta en alta dimensión — 'maldición de la "
            "dimensionalidad'). FAISS (aproximado): sub-lineal, O(log n) o mejor, "
            "a costa de exactitud — recomendado si n crece mucho más allá de ~30K."
        ),
        "notes": "Modelo más costoso en inferencia del set; candidato a correr en Colab si el Grid Search es exhaustivo.",
    },
    "naive_bayes": {
        "train": "O(n·p) — una sola pasada para estimar media/varianza por feature y clase.",
        "inference": "O(p·C) por consulta (C = número de clases; C=2 aquí).",
        "notes": (
            "GaussianNB soporta `partial_fit` para aprendizaje incremental "
            "(útil si el dataset no cupiera en memoria o llegara en streaming); "
            "no es necesario aquí (n~30K cabe en memoria), se menciona por completitud."
        ),
    },
    "logistic_regression": {
        "train": (
            "Depende del solver. SAGA (elegido en models.py): O(n·p) por época, "
            "con convergencia lineal gracias a reducción de varianza — escala mejor "
            "que solvers de segundo orden (ej. 'lbfgs'/'newton-cg', que aproximan la "
            "Hessiana y escalan peor con p) en datasets grandes como este (~30K filas)."
        ),
        "inference": "O(p) por consulta (producto punto con el vector de pesos).",
        "notes": "SAGA soporta L1, L2 y elasticnet — necesario aquí porque se optimiza `penalty`.",
    },
    "decision_tree": {
        "train": "O(n·p·log n) en promedio (ordenar features para evaluar cada split).",
        "inference": "O(log n) por consulta en promedio (profundidad del árbol).",
        "notes": "",
    },
    "random_forest": {
        "train": "O(T·n·p·log n) — T árboles independientes, cada uno O(n·p·log n).",
        "inference": "O(T·log n) por consulta (promedio sobre los T árboles).",
        "notes": "Paralelizable entre árboles (no usado aquí a nivel de n_jobs del modelo — ver models.py).",
    },
    "xgboost": {
        "train": (
            "O(T·n·p) con `tree_method='hist'` (histogramas: discretiza cada feature "
            "en ~256 bins, evaluando splits en O(n) por nivel) — bloque config del "
            "modelo. El método EXACTO (ordenar todas las features en cada split) es "
            "O(T·n·p·log n), notablemente más lento para n~30K."
        ),
        "inference": "O(T·log n) por consulta.",
        "notes": (
            "`early_stopping` con un set de validación independiente (no usado aquí: "
            "n_estimators se trata como hiperparámetro más dentro de la búsqueda, "
            "para no introducir un tercer nivel de partición además del outer/inner "
            "loop — simplificación consciente, documentada)."
        ),
    },
    "svm": {
        "train": (
            "O(n²) a O(n³) con kernel RBF completo (programación cuadrática sobre la "
            "matriz de kernel n×n) — el modelo con peor escalado teórico del set. "
            "Alternativas para n grande: LinearSVC (liblinear, sin kernel) O(n·p); "
            "SGDClassifier (hinge loss) O(n·p) por época — ambas evitan la matriz de "
            "kernel completa a costa de no capturar la no-linealidad de RBF."
        ),
        "inference": "O(s·p) por consulta (s = número de vectores de soporte, s ≤ n).",
        "notes": "Ver PROJECT_BRIEF: candidato más probable a correr en Colab si Grid Search+RBF es demasiado lento en local.",
    },
}


def _peak_memory_mb(fn, *args, **kwargs) -> tuple[Any, float, float]:
    """Ejecuta ``fn(*args, **kwargs)``, midiendo tiempo (s) y pico de
    memoria (MB) vía ``tracemalloc``.

    Returns
    -------
    result : lo que devuelva ``fn``
    elapsed_seconds : float
    peak_memory_mb : float
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 * 1024)


def measure_fit_predict(
    pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> dict[str, float]:
    """Mide tiempo y memoria de ``fit`` e ``predict`` para un pipeline dado.

    Parameters
    ----------
    pipeline : sklearn/imblearn Pipeline (sin ajustar)
    X_train, y_train, X_test : datos del fold recuperado (ver
        ``recover_outer_fold``)

    Returns
    -------
    dict con ``fit_time_seconds``, ``fit_peak_memory_mb``,
    ``predict_time_seconds``, ``predict_peak_memory_mb``,
    ``n_train``, ``n_test``, ``n_features_raw``.
    """
    _, fit_time, fit_peak_mb = _peak_memory_mb(pipeline.fit, X_train, y_train)
    y_pred, predict_time, predict_peak_mb = _peak_memory_mb(pipeline.predict, X_test)

    return {
        "fit_time_seconds": fit_time,
        "fit_peak_memory_mb": fit_peak_mb,
        "predict_time_seconds": predict_time,
        "predict_peak_memory_mb": predict_peak_mb,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features_raw": X_train.shape[1],
    }


def recover_outer_fold(
    X: pd.DataFrame,
    y: pd.Series,
    fold_index: int,
    n_outer_folds: int = config.N_OUTER_FOLDS,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Reconstruye el split exacto del fold externo ``fold_index`` de
    ``nested_cv.run_nested_cv``.

    ``NestedCVResult`` no guarda ``X_train``/``X_test`` por fold (para no
    duplicar el dataset completo en memoria/disco por cada una de las 104
    combinaciones). En su lugar, dado que ``StratifiedKFold`` con
    ``shuffle=True`` y una semilla fija es determinístico para un mismo
    ``(X, y)``, este helper reconstruye el MISMO split re-instanciando el
    splitter — siempre que se le pase el mismo ``(X, y)``, ``n_outer_folds``
    y ``random_state`` que usó la corrida original.

    Parameters
    ----------
    fold_index : int
        Índice del fold externo (0-indexado, en el orden que produce
        ``StratifiedKFold.split``).

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(islice(outer_cv.split(X, y), fold_index, fold_index + 1))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def select_best_fold_index(result: NestedCVResult, metric: str = "f1") -> int:
    """Elige el fold externo con mejor valor de ``metric`` — usado como
    representante "optimizado" en la tabla de complejidad (ver docstring
    de ``build_complexity_table`` para la justificación de por qué se
    necesita elegir UNO solo)."""
    best_fold = max(result.fold_results, key=lambda fr: fr.metrics[metric])
    return best_fold.fold_index


def compare_standard_vs_optimized(
    spec: ModelSpec,
    technique: str,
    best_params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    fold_index: int,
    n_outer_folds: int = config.N_OUTER_FOLDS,
    random_state: int = config.RANDOM_STATE,
    categorical_columns: list[str] | None = None,
    scoring: str = config.DEFAULT_SCORING,
) -> dict[str, Any]:
    """Compara el modelo ESTÁNDAR (hiperparámetros por defecto, sin
    balanceo) contra el OPTIMIZADO (``technique`` + ``best_params``
    encontrados por el pipeline de las 104 combinaciones) para un modelo,
    sobre el MISMO fold externo (para que la comparación de tiempo/memoria/
    desempeño sea sobre datos idénticos).

    Parameters
    ----------
    spec : ModelSpec
    technique : str
        Técnica de balanceo del modelo OPTIMIZADO (el estándar siempre usa
        ``"none"`` — sin balanceo, por definición de línea base).
    best_params : dict
        Hiperparámetros del modelo OPTIMIZADO para este fold (ver
        ``OuterFoldResult.best_params``).
    X, y : DataFrame, Series
        Dataset COMPLETO (mismo usado originalmente en
        ``nested_cv.run_nested_cv`` para esta combinación).
    fold_index : int
        Fold externo a usar (ver ``select_best_fold_index``).
    n_outer_folds, random_state : int
        Deben coincidir con los usados en la corrida original (ver
        ``recover_outer_fold``).
    categorical_columns : list[str], optional
    scoring : str, default=config.DEFAULT_SCORING

    Returns
    -------
    dict
        Una fila con complejidad teórica + tiempo/memoria/desempeño
        empírico de ambas variantes y el cambio en desempeño.
    """
    X_train, X_test, y_train, y_test = recover_outer_fold(
        X, y, fold_index, n_outer_folds=n_outer_folds, random_state=random_state
    )
    scorer = get_scorer(scoring)

    standard_steps = get_preprocessing_steps(spec, categorical_columns=categorical_columns)
    standard_pipeline = build_pipeline(
        spec, "none", y_train=y_train, preprocessing_steps=standard_steps, random_state=random_state
    )
    standard_perf = measure_fit_predict(standard_pipeline, X_train, y_train, X_test)
    standard_score = float(scorer(standard_pipeline, X_test, y_test))

    optimized_steps = get_preprocessing_steps(spec, categorical_columns=categorical_columns)
    optimized_pipeline = build_pipeline(
        spec,
        technique,
        y_train=y_train,
        preprocessing_steps=optimized_steps,
        random_state=random_state,
        **best_params,
    )
    optimized_perf = measure_fit_predict(optimized_pipeline, X_train, y_train, X_test)
    optimized_score = float(scorer(optimized_pipeline, X_test, y_test))

    complexity = TRAIN_INFERENCE_COMPLEXITY.get(spec.name, {})

    return {
        "model": spec.name,
        "theoretical_train_complexity": complexity.get("train", ""),
        "theoretical_inference_complexity": complexity.get("inference", ""),
        "complexity_notes": complexity.get("notes", ""),
        "fold_index": fold_index,
        "n_train": standard_perf["n_train"],
        "n_features_raw": standard_perf["n_features_raw"],
        "standard_fit_time_seconds": standard_perf["fit_time_seconds"],
        "standard_predict_time_seconds": standard_perf["predict_time_seconds"],
        "standard_fit_peak_memory_mb": standard_perf["fit_peak_memory_mb"],
        f"standard_{scoring}": standard_score,
        "optimized_technique": technique,
        "optimized_params": best_params,
        "optimized_fit_time_seconds": optimized_perf["fit_time_seconds"],
        "optimized_predict_time_seconds": optimized_perf["predict_time_seconds"],
        "optimized_fit_peak_memory_mb": optimized_perf["fit_peak_memory_mb"],
        f"optimized_{scoring}": optimized_score,
        "performance_change": optimized_score - standard_score,
    }


def build_complexity_table(
    selections: dict[str, tuple[str, NestedCVResult]],
    X: pd.DataFrame,
    y: pd.Series,
    n_outer_folds: int = config.N_OUTER_FOLDS,
    random_state: int = config.RANDOM_STATE,
    categorical_columns: list[str] | None = None,
    scoring: str = config.DEFAULT_SCORING,
    fold_selection_metric: str = "f1",
) -> pd.DataFrame:
    """Construye la tabla comparativa estándar vs. optimizado para varios modelos.

    Parameters
    ----------
    selections : dict[str, tuple[str, NestedCVResult]]
        ``{model_name: (technique, NestedCVResult)}`` — típicamente, para
        cada modelo, la combinación (técnica, método) con mejor desempeño
        en la tabla maestra de ``experiment_runner.py`` (bloque 6). Se pide
        UNA combinación por modelo (no las 104) porque la tabla de
        complejidad del curso es "por modelo", no por combinación.
    X, y : DataFrame, Series
        Dataset COMPLETO usado para generar esos ``NestedCVResult``.
    n_outer_folds, random_state : int
        Deben coincidir con los usados al generar los ``NestedCVResult``.
    fold_selection_metric : str, default="f1"
        Métrica usada para elegir, dentro de cada ``NestedCVResult``, cuál
        fold externo representa al modelo "optimizado" (ver
        ``select_best_fold_index``).

    Returns
    -------
    pd.DataFrame
        Una fila por modelo en ``selections``.
    """
    rows = []
    for model_name, (technique, result) in selections.items():
        spec = MODEL_REGISTRY[model_name]
        fold_index = select_best_fold_index(result, metric=fold_selection_metric)
        best_params = result.fold_results[fold_index].best_params
        row = compare_standard_vs_optimized(
            spec,
            technique,
            best_params,
            X,
            y,
            fold_index,
            n_outer_folds=n_outer_folds,
            random_state=random_state,
            categorical_columns=categorical_columns,
            scoring=scoring,
        )
        row["method"] = result.method
        rows.append(row)
    return pd.DataFrame(rows)
