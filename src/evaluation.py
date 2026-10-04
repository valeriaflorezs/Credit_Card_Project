"""Evaluación de modelos: matriz de confusión, curva ROC, tabla de
métricas (media±desviación estándar del outer loop), calibración
(reliability diagram, Brier, ECE, recalibración) y análisis de
sensibilidad a la semilla aleatoria.

**Justificación de la métrica principal de validación**

El dataset tiene un desbalance ~78/22 (clase mayoritaria = no incumple).
En ese escenario, **accuracy es engañosa**: un clasificador trivial que
siempre predice "no incumple" ya obtiene ~78% de accuracy sin aprender
nada — es precisamente el problema que el profesor señaló al rechazar el
0.93 de accuracy del Entregable 2 (ver PROJECT_BRIEF.md).

En el contexto de riesgo crediticio, los dos tipos de error tienen costos
muy distintos:

- **Falso negativo** (predecir "no incumple" cuando SÍ incumple): el banco
  otorga o mantiene crédito a un cliente que terminará en default — pierde
  el principal más los intereses no pagados. Costo alto.
- **Falso positivo** (predecir "incumple" cuando NO incumple): el banco
  trata con más cautela (o niega/limita crédito) a un cliente que en
  realidad era solvente — costo de oportunidad (ingresos por intereses no
  percibidos), pero no hay pérdida de principal.

Esta asimetría (FN típicamente más costoso que FP) favorece **recall**
(minimizar falsos negativos) o, si se quiere balancear con la precisión
para no inundar de falsos positivos costosos en revisión manual, **F1**
(media armónica de precision/recall). Por eso ``config.DEFAULT_SCORING =
"f1"`` — un compromiso razonable por defecto — y por eso este módulo
reporta las 5 métricas núcleo (nunca solo accuracy) y expone
``expected_cost()`` para cuantificar la elección en términos de costo
explícito, no solo argumento cualitativo. La proporción real FN/FP debe
calibrarse con el equipo/negocio; aquí se ilustra con un supuesto
razonable (5:1), declarado explícitamente como supuesto, no como dato.

**División de responsabilidades:** este módulo NUNCA reajusta un modelo
para calcular matriz de confusión / ROC / métricas núcleo — usa
directamente ``y_true``/``y_pred``/``y_proba`` ya guardados por
``nested_cv.OuterFoldResult``. Sí reajusta pipelines (una vez, sobre datos
ya recuperados con ``complexity_analysis.recover_outer_fold``) para la
recalibración y el análisis de sensibilidad a semillas, que por
definición requieren entrenar variantes adicionales.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from src import config
from src.balancing import build_pipeline
from src.complexity_analysis import recover_outer_fold
from src.models import ModelSpec
from src.nested_cv import NestedCVResult, compute_core_metrics
from src.preprocessing import get_preprocessing_steps

SEED_SENSITIVE_MODELS: tuple[str, ...] = ("random_forest", "xgboost")


def expected_cost(
    y_true: np.ndarray, y_pred: np.ndarray, cost_fn: float = 5.0, cost_fp: float = 1.0
) -> dict[str, float]:
    """Costo esperado ponderado por falsos negativos/positivos.

    ``cost_fn``/``cost_fp`` son un SUPUESTO ILUSTRATIVO (5:1 — un default
    no detectado cuesta ~5x lo que un cliente solvente tratado con más
    cautela), no una cifra derivada de datos reales del banco — ajustar
    con el costo real si se dispone de él. El punto es cuantificar la
    justificación de preferir recall/F1 sobre accuracy, no dar una cifra
    definitiva.

    Returns
    -------
    dict con conteos de la matriz de confusión, costo total y costo por
    observación.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_cost = fn * cost_fn + fp * cost_fp
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "total_cost": float(total_cost),
        "cost_per_sample": float(total_cost / len(y_true)),
    }


# ---------------------------------------------------------------------------
# Matriz de confusión / curva ROC — agregadas out-of-fold
# ---------------------------------------------------------------------------


def aggregate_confusion_matrix(result: NestedCVResult) -> np.ndarray:
    """Suma las matrices de confusión de los folds del outer loop.

    Como cada observación del dataset aparece exactamente una vez como
    fold de PRUEBA a través de los folds externos (partición de
    ``StratifiedKFold``), sumar las matrices de confusión por fold
    equivale a la matriz de confusión "out-of-fold" sobre el dataset
    completo — más estable que mirar un solo fold.

    Returns
    -------
    np.ndarray de forma (2, 2): [[TN, FP], [FN, TP]].
    """
    cm_total = np.zeros((2, 2), dtype=int)
    for fr in result.fold_results:
        cm_total += confusion_matrix(fr.y_true, fr.y_pred, labels=[0, 1])
    return cm_total


def pooled_predictions(result: NestedCVResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatena ``y_true``/``y_pred``/``y_proba`` de TODOS los folds
    externos (predicciones out-of-fold), para curvas/diagramas agregados.
    """
    y_true = np.concatenate([fr.y_true for fr in result.fold_results])
    y_pred = np.concatenate([fr.y_pred for fr in result.fold_results])
    y_proba = np.concatenate([fr.y_proba for fr in result.fold_results])
    return y_true, y_pred, y_proba


def regenerate_outof_fold_predictions(
    spec: ModelSpec,
    technique: str,
    best_params_per_fold: list[dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    n_outer_folds: int = config.N_OUTER_FOLDS,
    random_state: int = config.RANDOM_STATE,
    categorical_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruye ``y_true``/``y_pred``/``y_proba`` out-of-fold (ver
    ``pooled_predictions``) SIN volver a correr la búsqueda de
    hiperparámetros — a partir de ``best_params_per_fold`` ya encontrados
    y guardados en la tabla maestra (columna JSON
    ``"best_params_per_fold"``, ver ``nested_cv.NestedCVResult.summary_row``).

    Necesario porque un ``NestedCVResult`` (con sus ``y_true``/``y_pred``/
    ``y_proba`` por fold) vive solo en memoria durante la sesión que corrió
    ``experiment_runner.run_experiments`` — la tabla maestra en CSV no
    guarda esos arrays (serían enormes y no caben bien en una fila). Una
    sesión NUEVA (ej. ``05_results_analysis.ipynb``, abierto después de
    ``04_run_experiments.ipynb``) solo tiene el CSV — este helper recupera
    cada fold (``complexity_analysis.recover_outer_fold``) y REAJUSTA el
    pipeline con los ``best_params`` YA CONOCIDOS de ese fold (barato: un
    solo ``fit``, no una búsqueda), en vez de reoptimizar desde cero.

    Parameters
    ----------
    best_params_per_fold : list[dict]
        Un dict de hiperparámetros por fold externo, en orden (ver
        columna ``best_params_per_fold`` de la tabla maestra — deserializar
        con ``json.loads`` antes de pasarlo aquí).

    Returns
    -------
    y_true, y_pred, y_proba : np.ndarray
        Concatenados en el mismo orden que produciría
        ``nested_cv.run_nested_cv`` + ``pooled_predictions``.
    """
    preprocessing_steps = get_preprocessing_steps(spec, categorical_columns=categorical_columns)

    y_true_parts, y_pred_parts, y_proba_parts = [], [], []
    for fold_index, best_params in enumerate(best_params_per_fold):
        X_train, X_test, y_train, y_test = recover_outer_fold(
            X, y, fold_index, n_outer_folds=n_outer_folds, random_state=random_state
        )
        pipeline = build_pipeline(
            spec,
            technique,
            y_train=y_train,
            preprocessing_steps=preprocessing_steps,
            random_state=random_state,
            **best_params,
        )
        pipeline.fit(X_train, y_train)
        y_true_parts.append(y_test.to_numpy())
        y_pred_parts.append(pipeline.predict(X_test))
        y_proba_parts.append(pipeline.predict_proba(X_test)[:, 1])

    return np.concatenate(y_true_parts), np.concatenate(y_pred_parts), np.concatenate(y_proba_parts)


def pooled_roc_curve(result: NestedCVResult) -> dict[str, Any]:
    """Curva ROC sobre las predicciones out-of-fold agregadas (ver
    ``pooled_predictions``) — un solo AUC/curva por combinación, en vez de
    una curva ruidosa por cada fold pequeño.
    """
    y_true, _, y_proba = pooled_predictions(result)
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": float(roc_auc_score(y_true, y_proba)),
    }


# ---------------------------------------------------------------------------
# Tabla de métricas — media ± desviación estándar del outer loop, NUNCA un
# valor puntual de una sola partición
# ---------------------------------------------------------------------------


def format_metrics_table(master_table: pd.DataFrame, metrics: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Formatea la tabla maestra (``experiment_runner.run_experiments``)
    con columnas ``"{metrica}: mean ± std"`` legibles para el reporte, sin
    perder las columnas numéricas originales.

    Parameters
    ----------
    master_table : pd.DataFrame
        Con columnas ``{metrica}_mean``/``{metrica}_std`` (ver
        ``nested_cv.NestedCVResult.summary_row``).
    metrics : tuple[str], optional
        Por defecto, ``nested_cv.CORE_METRICS``.
    """
    from src.nested_cv import CORE_METRICS

    metrics = metrics or CORE_METRICS
    display_table = master_table[["model", "technique", "method"]].copy()
    for metric_name in metrics:
        mean_col, std_col = f"{metric_name}_mean", f"{metric_name}_std"
        if mean_col in master_table.columns and std_col in master_table.columns:
            display_table[metric_name] = (
                master_table[mean_col].round(4).astype(str) + " ± " + master_table[std_col].round(4).astype(str)
            )
    return display_table


# ---------------------------------------------------------------------------
# Calibración: reliability diagram, Brier, ECE, recalibración
# ---------------------------------------------------------------------------


def _expected_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> float:
    """ECE = sum_b (n_b / N) * |accuracy_b - confianza_b|.

    No está en scikit-learn directamente (``calibration_curve`` da los
    puntos del reliability diagram pero no agrega el error ponderado por
    tamaño de bin) — implementación directa de la definición estándar.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.quantile(y_proba, np.linspace(0.0, 1.0, n_bins + 1))
        bin_edges[0], bin_edges[-1] = 0.0, 1.0
    else:
        raise ValueError(f"strategy debe ser 'uniform' o 'quantile', no '{strategy}'")

    bin_ids = np.clip(np.digitize(y_proba, bin_edges[1:-1], right=True), 0, n_bins - 1)
    n_total = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        accuracy_b = float(np.mean(y_true[mask]))
        confidence_b = float(np.mean(y_proba[mask]))
        ece += (np.sum(mask) / n_total) * abs(accuracy_b - confidence_b)
    return float(ece)


def evaluate_calibration(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> dict[str, Any]:
    """Brier Score + ECE + puntos del reliability diagram.

    Returns
    -------
    dict con ``brier_score``, ``ece``, ``reliability_curve``
    (``{"prob_true": ..., "prob_pred": ...}``, listo para graficar), y los
    parámetros usados (``n_bins``, ``strategy``).
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy=strategy)
    return {
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "ece": _expected_calibration_error(y_true, y_proba, n_bins=n_bins, strategy=strategy),
        "reliability_curve": {"prob_true": prob_true, "prob_pred": prob_pred},
        "n_bins": n_bins,
        "strategy": strategy,
    }


def pooled_calibration(result: NestedCVResult, n_bins: int = 10, strategy: str = "uniform") -> dict[str, Any]:
    """``evaluate_calibration`` sobre las predicciones out-of-fold agregadas
    de las 3 (o N) particiones del outer loop — igual razón que
    ``pooled_roc_curve``: más estable que un solo fold pequeño.
    """
    y_true, _, y_proba = pooled_predictions(result)
    return evaluate_calibration(y_true, y_proba, n_bins=n_bins, strategy=strategy)


def is_calibration_poor(ece: float, threshold: float = 0.05) -> bool:
    """Heurística simple: ECE > ``threshold`` sugiere recalibrar.

    El umbral (0.05 = 5 puntos porcentuales de discrepancia media entre
    confianza y frecuencia observada) es una convención común en la
    literatura de calibración, no una ley — documentado como heurística.
    """
    return ece > threshold


def fit_recalibrator(
    y_calib: np.ndarray, proba_calib: np.ndarray, method: Literal["sigmoid", "isotonic"] = "sigmoid"
) -> Callable[[np.ndarray], np.ndarray]:
    """Ajusta un recalibrador de probabilidades (Platt scaling o isotónica).

    Implementado con primitivas estables de scikit-learn en vez de
    ``CalibratedClassifierCV(cv="prefit")`` (API de "prefit" deprecada/
    cambiante entre versiones recientes de scikit-learn) — Platt scaling
    ES, por definición, una regresión logística 1D sobre el score crudo, y
    la recalibración isotónica ES ``IsotonicRegression`` — esta
    implementación es matemáticamente idéntica y no depende de una API que
    puede romperse con la versión de scikit-learn instalada.

    Parameters
    ----------
    y_calib, proba_calib : array-like
        Etiquetas verdaderas y probabilidades CRUDAS (del modelo ya
        entrenado) de un conjunto de CALIBRACIÓN separado del de
        entrenamiento (ver ``assess_and_recalibrate``) — nunca el mismo
        conjunto usado para entrenar el modelo, o la recalibración
        sobreajustaría.
    method : {"sigmoid", "isotonic"}
        ``"sigmoid"`` = Platt scaling (paramétrico, mejor con pocos datos
        de calibración). ``"isotonic"`` = no paramétrico (más flexible,
        necesita más datos de calibración para no sobreajustar).

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Función que mapea probabilidades crudas -> recalibradas.
    """
    proba_calib = np.asarray(proba_calib).reshape(-1, 1)
    y_calib = np.asarray(y_calib)

    if method == "sigmoid":
        platt = LogisticRegression()
        platt.fit(proba_calib, y_calib)
        return lambda p: platt.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(proba_calib.ravel(), y_calib)
        return lambda p: isotonic.predict(np.asarray(p))
    raise ValueError(f"method debe ser 'sigmoid' o 'isotonic', no '{method}'")


def assess_and_recalibrate(
    spec: ModelSpec,
    technique: str,
    best_params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    fold_index: int,
    method: Literal["sigmoid", "isotonic"] = "sigmoid",
    calib_fraction: float = 0.3,
    n_bins: int = 10,
    n_outer_folds: int = config.N_OUTER_FOLDS,
    random_state: int = config.RANDOM_STATE,
    categorical_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Evalúa la calibración de un modelo y reporta el efecto de recalibrarlo.

    Reconstruye el fold externo (ver ``complexity_analysis.recover_outer_fold``),
    reserva una porción del fold de entrenamiento como conjunto de
    CALIBRACIÓN (nunca visto por el modelo durante su propio ``fit``, para
    que la recalibración no esté contaminada por sobreajuste), entrena el
    pipeline sobre el resto, y compara Brier/ECE antes y después de
    recalibrar sobre el fold de PRUEBA externo.

    Parameters
    ----------
    fold_index : int
        Ver ``complexity_analysis.select_best_fold_index``.
    calib_fraction : float, default=0.3
        Proporción del fold de entrenamiento externo reservada para
        calibración (nunca usada para entrenar el modelo).

    Returns
    -------
    dict con ``before`` / ``after`` (salida de ``evaluate_calibration``
    sobre el fold de prueba externo, antes/después de recalibrar) y
    ``recalibration_applied`` (bool, según ``is_calibration_poor``).
    """
    X_train_outer, X_test_outer, y_train_outer, y_test_outer = recover_outer_fold(
        X, y, fold_index, n_outer_folds=n_outer_folds, random_state=random_state
    )
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train_outer,
        y_train_outer,
        test_size=calib_fraction,
        stratify=y_train_outer,
        random_state=random_state,
    )

    preprocessing_steps = get_preprocessing_steps(spec, categorical_columns=categorical_columns)
    pipeline = build_pipeline(
        spec,
        technique,
        y_train=y_fit,
        preprocessing_steps=preprocessing_steps,
        random_state=random_state,
        **best_params,
    )
    pipeline.fit(X_fit, y_fit)

    proba_calib = pipeline.predict_proba(X_calib)[:, 1]
    proba_test_raw = pipeline.predict_proba(X_test_outer)[:, 1]

    before = evaluate_calibration(y_test_outer.to_numpy(), proba_test_raw, n_bins=n_bins)
    recalibration_applied = is_calibration_poor(before["ece"])

    recalibrator = fit_recalibrator(y_calib, proba_calib, method=method)
    proba_test_recalibrated = recalibrator(proba_test_raw)
    after = evaluate_calibration(y_test_outer.to_numpy(), proba_test_recalibrated, n_bins=n_bins)

    return {
        "model": spec.name,
        "technique": technique,
        "method_recalibration": method,
        "fold_index": fold_index,
        "before": before,
        "after": after,
        "recalibration_recommended": recalibration_applied,
        "brier_improvement": before["brier_score"] - after["brier_score"],
        "ece_improvement": before["ece"] - after["ece"],
    }


# ---------------------------------------------------------------------------
# Sensibilidad a la semilla aleatoria (bloque 10, sección "robustez")
# ---------------------------------------------------------------------------


def seed_sensitivity_analysis(
    spec: ModelSpec,
    technique: str,
    best_params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    fold_index: int,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    n_outer_folds: int = config.N_OUTER_FOLDS,
    base_random_state: int = config.RANDOM_STATE,
    categorical_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Reentrena el mismo modelo/hiperparámetros con distintas semillas
    aleatorias INTERNAS, sobre el MISMO fold, para medir cuánto varía el
    desempeño por azar de inicialización — separado de la varianza entre
    folds que ya reporta ``nested_cv.NestedCVResult.metrics_summary``.

    Solo tiene sentido para modelos con aleatoriedad propia en el
    entrenamiento — ver ``SEED_SENSITIVE_MODELS`` (Random Forest:
    bootstrap de muestras/features; XGBoost: subsample/colsample
    estocásticos). Para modelos determinísticos dado un mismo
    hiperparámetro (KNN, Naive Bayes, Regresión Logística con solver
    determinista, Decision Tree, SVM) se puede correr igual como chequeo
    de que la varianza entre semillas es ~0 (confirma que el modelo es,
    en efecto, determinista).

    Nota importante: el fold de datos (``fold_index``) se mantiene fijo —
    la partición de ``StratifiedKFold`` (``n_outer_folds``,
    ``base_random_state``) NO cambia con ``seeds``, solo la semilla interna
    del ESTIMADOR. Cambiar ambas a la vez confundiría "sensibilidad a la
    semilla del modelo" con "variabilidad entre folds".

    Returns
    -------
    pd.DataFrame
        Una fila por semilla, con las métricas núcleo de
        ``nested_cv.CORE_METRICS``.
    """
    if spec.name not in SEED_SENSITIVE_MODELS:
        import warnings

        warnings.warn(
            f"'{spec.name}' no está en SEED_SENSITIVE_MODELS ({SEED_SENSITIVE_MODELS}); "
            "se espera varianza ~0 entre semillas (modelo determinista dado el hiperparámetro).",
            stacklevel=2,
        )

    X_train_outer, X_test_outer, y_train_outer, y_test_outer = recover_outer_fold(
        X, y, fold_index, n_outer_folds=n_outer_folds, random_state=base_random_state
    )
    preprocessing_steps = get_preprocessing_steps(spec, categorical_columns=categorical_columns)

    rows = []
    for seed in seeds:
        pipeline = build_pipeline(
            spec,
            technique,
            y_train=y_train_outer,
            preprocessing_steps=preprocessing_steps,
            random_state=seed,
            **best_params,
        )
        pipeline.fit(X_train_outer, y_train_outer)
        y_pred = pipeline.predict(X_test_outer)
        y_proba = pipeline.predict_proba(X_test_outer)[:, 1]
        metrics = compute_core_metrics(y_test_outer.to_numpy(), y_pred, y_proba)
        rows.append({"seed": seed, **metrics})

    return pd.DataFrame(rows)


def summarize_seed_sensitivity(seed_results: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Media ± desviación estándar de cada métrica a través de las semillas
    de ``seed_sensitivity_analysis`` — comparar esta desviación estándar
    contra la de ``NestedCVResult.metrics_summary`` (entre folds) es lo que
    motiva la pregunta de la sección 10: ¿la diferencia entre modelos es
    mayor que su propia variabilidad intrínseca? (Se responde formalmente
    en ``stats_comparison.py``, bloque 10.)
    """
    metric_cols = [c for c in seed_results.columns if c != "seed"]
    return {col: (float(seed_results[col].mean()), float(seed_results[col].std())) for col in metric_cols}
