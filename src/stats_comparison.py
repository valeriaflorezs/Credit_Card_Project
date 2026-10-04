"""Comparación estadística jerárquica de las combinaciones (modelo ×
balanceo × optimización): Friedman (ómnibus) -> Nemenyi + CD diagram
(post-hoc, SOLO si Friedman es significativo) -> DeLong sobre un
subconjunto reducido (top 2-3), con corrección por comparaciones
múltiples -> Cliff's Delta (tamaño del efecto) -> bootstrap BCa (IC).

**Por qué jerárquico y no todas las comparaciones pareadas:** con ~104
combinaciones válidas hay C(104, 2) = 5,356 pares posibles. Correr DeLong
(u otro test) sobre todos ellos sin corrección infla el error tipo I —
cientos de "diferencias significativas" solo por azar. Por eso: primero
Friedman responde si HAY evidencia de que no todas las combinaciones son
equivalentes; solo si la respuesta es sí se identifica CUÁLES difieren
(Nemenyi + CD diagram) y, para un subconjunto reducido y justificado
(las 2-3 mejores por rango), se comparan sus curvas ROC directamente
(DeLong) con corrección de Holm-Bonferroni/Benjamini-Hochberg.

**Limitación honesta de los datos:** ``config.N_OUTER_FOLDS = 3`` (elegido
por costo computacional dado el volumen de 104 combinaciones — ver
config.py) significa que Friedman/Nemenyi comparan solo 3 "bloques", muy
por debajo de lo recomendado en la literatura (Demšar 2006 sugiere ~10+
datasets/folds para potencia estadística razonable). Este módulo reporta
los resultados igual — es la validación cruzada con la que se generaron
las 104 corridas — pero si Friedman NO resulta significativo, la
conclusión correcta con tan pocos bloques es "no hay potencia suficiente
para detectar diferencias", no necesariamente "los modelos son
equivalentes". Esta ambigüedad debe discutirse explícitamente en el
reporte, no ocultarse.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Callable

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, norm, studentized_range
from sklearn.metrics import f1_score
from statsmodels.stats.multitest import multipletests

from src.evaluation import pooled_predictions, regenerate_outof_fold_predictions
from src.nested_cv import NestedCVResult

ResultsDict = dict[tuple[str, str, str], NestedCVResult]


def _combo_label(model: str, technique: str, method: str) -> str:
    return f"{model}|{technique}|{method}"


# ---------------------------------------------------------------------------
# Etapa 1 — Friedman (ómnibus)
# ---------------------------------------------------------------------------


def build_performance_matrix(results: ResultsDict, metric: str = "f1") -> pd.DataFrame:
    """Matriz de bloque (folds del outer loop × combinaciones) para Friedman/Nemenyi.

    Parameters
    ----------
    results : dict[(model, technique, method), NestedCVResult]
        Ej. la salida de ``experiment_runner.run_experiments``.
    metric : str, default="f1"
        Una de ``nested_cv.CORE_METRICS``.

    Returns
    -------
    pd.DataFrame
        Filas = índice de fold externo, columnas = ``"modelo|tecnica|metodo"``.
        Todas las combinaciones DEBEN tener el mismo número de folds
        externos (garantizado si todas se corrieron con el mismo
        ``n_outer_folds`` — el caso normal de ``experiment_runner.py``).
    """
    n_folds = {len(r.fold_results) for r in results.values()}
    if len(n_folds) > 1:
        raise ValueError(
            f"Todas las combinaciones deben tener el mismo número de outer folds para Friedman; se encontraron: {n_folds}"
        )

    data = {
        _combo_label(model, technique, method): [fr.metrics[metric] for fr in result.fold_results]
        for (model, technique, method), result in results.items()
    }
    return pd.DataFrame(data)


def build_performance_matrix_from_table(master_table: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Igual que ``build_performance_matrix``, pero a partir de la tabla
    maestra en CSV (``experiment_runner.load_master_table``) en vez de un
    dict de ``NestedCVResult`` en memoria.

    Necesario para correr la comparación jerárquica en una sesión NUEVA
    (ej. ``05_results_analysis.ipynb``, después de que
    ``04_run_experiments.ipynb`` ya haya guardado y cerrado) — usa la
    columna ``"{metric}_per_fold"`` (JSON) que
    ``nested_cv.NestedCVResult.summary_row`` persiste para este propósito
    exacto.

    Parameters
    ----------
    master_table : pd.DataFrame
        Con columnas ``model``, ``technique``, ``method`` y
        ``"{metric}_per_fold"``.
    metric : str, default="f1"

    Returns
    -------
    pd.DataFrame
        Mismo formato que ``build_performance_matrix``.
    """
    import json

    col = f"{metric}_per_fold"
    if col not in master_table.columns:
        raise KeyError(
            f"La tabla maestra no tiene la columna '{col}'. Si viene de una corrida antigua "
            "(antes de que summary_row() guardara valores por fold), hay que re-correr esas "
            "combinaciones con experiment_runner.run_experiments()."
        )

    data = {}
    for _, row in master_table.iterrows():
        label = _combo_label(row["model"], row["technique"], row["method"])
        data[label] = json.loads(row[col])
    return pd.DataFrame(data)


def friedman_test(performance_matrix: pd.DataFrame, alpha: float = 0.05) -> dict[str, Any]:
    """Test de Friedman: ¿los rangos promedio de las combinaciones difieren
    más de lo esperado por azar?

    H0: todas las combinaciones tienen rendimiento equivalente (rangos
    promedio iguales). Si no se rechaza H0, se reporta así explícitamente
    — NO se procede a Nemenyi/CD diagram/DeLong (ver docstring del módulo).

    Returns
    -------
    dict con ``statistic``, ``p_value``, ``significant`` (bool),
    ``avg_ranks`` (Series ordenada, rank 1 = mejor), ``n_blocks``, ``k_combinations``.
    """
    columns = list(performance_matrix.columns)
    samples = [performance_matrix[col].to_numpy() for col in columns]
    statistic, p_value = friedmanchisquare(*samples)

    # Rank 1 = mejor desempeño (ascending=False sobre la métrica de performance).
    ranks = performance_matrix.rank(axis=1, ascending=False, method="average")
    avg_ranks = ranks.mean(axis=0).sort_values()

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "avg_ranks": avg_ranks,
        "n_blocks": performance_matrix.shape[0],
        "k_combinations": performance_matrix.shape[1],
    }


# ---------------------------------------------------------------------------
# Etapa 2a — Nemenyi post-hoc + CD diagram (solo si Friedman es significativo)
# ---------------------------------------------------------------------------


def nemenyi_posthoc(performance_matrix: pd.DataFrame) -> pd.DataFrame:
    """Matriz de p-valores pareados de Nemenyi (ya corrige el error familywise
    conjunto vía la distribución del rango studentizado — no requiere
    corrección adicional tipo Holm/BH sobre esta matriz).

    Returns
    -------
    pd.DataFrame (k × k), índice/columnas = etiquetas de ``performance_matrix``.
    """
    return sp.posthoc_nemenyi_friedman(performance_matrix)


def compute_critical_difference(k: int, n_blocks: int, alpha: float = 0.05) -> float:
    """Diferencia Crítica (CD) de Nemenyi: ``CD = q_alpha * sqrt(k(k+1)/(6N))``.

    ``q_alpha`` se deriva de la distribución del rango studentizado con
    ``df=inf`` (supuesto estándar de Nemenyi) dividido por ``sqrt(2)`` —
    verificado contra la tabla de Demšar (2006): para k=5, alpha=0.05 da
    q_alpha≈2.728, coincidiendo con el valor tabulado.

    Parameters
    ----------
    k : int
        Número de combinaciones comparadas.
    n_blocks : int
        Número de folds externos (bloques).
    """
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_blocks)))


def plot_cd_diagram(
    avg_ranks: pd.Series,
    sig_matrix: pd.DataFrame,
    cd: float | None = None,
    alpha: float = 0.05,
    save_path: str | None = None,
):
    """Diagrama de Diferencia Crítica — sustituye a una tabla de miles de
    comparaciones (obligatorio según el curso si Friedman es significativo).

    Wrapper sobre ``scikit_posthocs.critical_difference_diagram``.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(avg_ranks))))
    sp.critical_difference_diagram(avg_ranks, sig_matrix, cd=cd, alpha=alpha, ax=ax)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Etapa 2b — DeLong (subconjunto reducido) + corrección múltiple
# ---------------------------------------------------------------------------


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks (promedia rangos empatados) — paso interno del algoritmo
    rápido de DeLong (Sun & Xu, 2014)."""
    sorted_idx = np.argsort(x, kind="mergesort")
    x_sorted = x[sorted_idx]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[sorted_idx] = ranks
    return out


def _fast_delong_covariance(predictions: np.ndarray, n_positive: int) -> tuple[np.ndarray, np.ndarray]:
    """Algoritmo rápido de DeLong (Sun & Xu, 2014): AUCs + matriz de
    covarianza entre las AUCs de ``k`` clasificadores sobre el MISMO
    conjunto de prueba (por eso la comparación debe tratar la correlación
    entre curvas — un t-test pareado ingenuo la ignoraría).

    Parameters
    ----------
    predictions : np.ndarray, forma (k, n_total)
        Scores de cada clasificador, columnas ordenadas con los ``n_positive``
        positivos PRIMERO.
    n_positive : int

    Returns
    -------
    aucs : np.ndarray, forma (k,)
    delong_cov : np.ndarray, forma (k, k)
    """
    m = n_positive
    n = predictions.shape[1] - m
    k = predictions.shape[0]

    positive_examples = predictions[:, :m]
    negative_examples = predictions[:, m:]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = np.atleast_2d(sx) / m + np.atleast_2d(sy) / n
    return aucs, delong_cov


def delong_test(y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray) -> dict[str, float]:
    """Test de DeLong: compara dos AUCs CORRELACIONADAS (mismo conjunto de
    prueba, ver justificación en ``_fast_delong_covariance``).

    H0: AUC_a == AUC_b.

    Returns
    -------
    dict con ``auc_a``, ``auc_b``, ``z_statistic``, ``p_value``.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true, kind="mergesort")  # positivos (1) primero
    n_positive = int(y_true.sum())

    predictions = np.vstack([np.asarray(proba_a), np.asarray(proba_b)])[:, order]
    aucs, delong_cov = _fast_delong_covariance(predictions, n_positive)

    auc_diff = aucs[0] - aucs[1]
    variance = delong_cov[0, 0] + delong_cov[1, 1] - 2 * delong_cov[0, 1]
    if variance <= 0:
        z_statistic, p_value = 0.0, 1.0
    else:
        z_statistic = float(auc_diff / np.sqrt(variance))
        p_value = float(2 * (1 - norm.cdf(abs(z_statistic))))

    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z_statistic": z_statistic, "p_value": p_value}


def delong_comparisons_from_pooled(
    pooled: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    multiple_test_method: str = "holm",
) -> pd.DataFrame:
    """Compara TODOS los pares de un subconjunto reducido (top 2-3
    combinaciones) con DeLong, y corrige por comparaciones múltiples
    (Holm-Bonferroni o Benjamini-Hochberg/FDR).

    Función de bajo nivel, agnóstica a de DÓNDE vienen las predicciones
    out-of-fold pooleadas — acepta cualquier dict ``{etiqueta: (y_true,
    y_pred, y_proba)}``, ya sea leído directamente de un
    ``NestedCVResult`` en memoria (ver ``compare_top_combinations_delong``)
    o RECONSTRUIDO desde la tabla maestra en una sesión nueva (ver
    ``evaluation.regenerate_outof_fold_predictions`` y
    ``run_hierarchical_comparison_from_table``).

    Parameters
    ----------
    pooled : dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
        2 o 3 entradas ``{etiqueta: (y_true, y_pred, y_proba)}``. Todas
        deben compartir el mismo ``y_true`` (mismo dataset/particiones) —
        condición que exige DeLong para que la comparación sea válida.
    multiple_test_method : {"holm", "fdr_bh"}, default="holm"
        Ver ``statsmodels.stats.multitest.multipletests``.

    Returns
    -------
    pd.DataFrame
        Una fila por par, con ``p_value`` (crudo) y ``p_value_adjusted``.
    """
    labels = list(pooled.keys())
    if len(labels) < 2:
        raise ValueError("Se necesitan al menos 2 combinaciones para comparar con DeLong.")

    rows = []
    for label_a, label_b in combinations(labels, 2):
        y_true_a, _, proba_a = pooled[label_a]
        y_true_b, _, proba_b = pooled[label_b]
        if not np.array_equal(y_true_a, y_true_b):
            raise ValueError(
                "Las predicciones out-of-fold de ambas combinaciones deben provenir del MISMO "
                "dataset/particiones (mismo X, y, n_outer_folds, random_state) para que DeLong sea válido."
            )
        test_result = delong_test(y_true_a, proba_a, proba_b)
        rows.append({"combo_a": label_a, "combo_b": label_b, **test_result})

    df = pd.DataFrame(rows)
    _, adjusted, _, _ = multipletests(df["p_value"], method=multiple_test_method)
    df["p_value_adjusted"] = adjusted
    df["multiple_test_method"] = multiple_test_method
    return df


def compare_top_combinations_delong(
    results: ResultsDict,
    top_combos: list[tuple[str, str, str]],
    multiple_test_method: str = "holm",
) -> pd.DataFrame:
    """Como ``delong_comparisons_from_pooled``, pero a partir de un dict de
    ``NestedCVResult`` EN MEMORIA (ej. la salida de
    ``experiment_runner.run_experiments`` en la misma sesión).

    Parameters
    ----------
    top_combos : list[tuple[str, str, str]]
        2 o 3 combinaciones (model, technique, method) — ver
        ``friedman_test``'s ``avg_ranks`` para elegir las de mejor rango.
    multiple_test_method : {"holm", "fdr_bh"}, default="holm"
        Ver ``statsmodels.stats.multitest.multipletests``.

    Returns
    -------
    pd.DataFrame
        Una fila por par, con ``p_value`` (crudo) y ``p_value_adjusted``.
    """
    pooled = {_combo_label(*combo): pooled_predictions(results[combo]) for combo in top_combos}
    return delong_comparisons_from_pooled(pooled, multiple_test_method=multiple_test_method)


# ---------------------------------------------------------------------------
# Tamaño del efecto — Cliff's Delta
# ---------------------------------------------------------------------------


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's Delta: tamaño del efecto no paramétrico basado en dominancia
    de rangos (consistente con el enfoque de Friedman/Nemenyi, que también
    usa rangos en vez de valores puntuales).

    delta = (P(a > b) - P(a < b)), en [-1, 1]. 0 = superposición total,
    ±1 = separación total.

    Nota: con ``config.N_OUTER_FOLDS = 3`` valores por combinación, esta
    estimación es extremadamente burda (solo 9 pares posibles entre dos
    combinaciones) — reportar junto con la limitación de potencia
    documentada en el docstring del módulo, no como una cifra definitiva.
    """
    a, b = np.asarray(a), np.asarray(b)
    greater = np.sum(a[:, None] > b[None, :])
    less = np.sum(a[:, None] < b[None, :])
    return float((greater - less) / (len(a) * len(b)))


def interpret_cliffs_delta(delta: float) -> str:
    """Umbrales convencionales (Romano et al., 2006): |d|<0.147 despreciable,
    <0.33 pequeño, <0.474 mediano, si no, grande."""
    d = abs(delta)
    if d < 0.147:
        return "despreciable"
    if d < 0.33:
        return "pequeño"
    if d < 0.474:
        return "mediano"
    return "grande"


# ---------------------------------------------------------------------------
# Bootstrap BCa — intervalos de confianza
# ---------------------------------------------------------------------------


def bootstrap_bca_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = f1_score,
    confidence_level: float = 0.95,
    n_resamples: int = 2000,
    random_state: int | None = None,
) -> dict[str, float]:
    """Intervalo de confianza bootstrap BCa (bias-corrected and accelerated)
    para una métrica de desempeño, sobre las predicciones out-of-fold
    pooleadas (ver ``evaluation.pooled_predictions``) — no sobre los 3
    valores por fold (demasiado pocos para un bootstrap razonable).

    Implementación manual (no hay una función BCa lista para métricas de
    clasificación arbitrarias en scipy/sklearn): remuestrea pares
    (y_true_i, y_score_i) con reemplazo, corrige el sesgo (``z0``) y la
    aceleración (``a``, vía jackknife) siguiendo Efron & Tibshirani (1993).

    Parameters
    ----------
    metric_fn : callable(y_true, y_score) -> float
        Ej. ``sklearn.metrics.f1_score`` (con ``y_score`` como predicciones
        binarias) o ``roc_auc_score`` (con ``y_score`` como probabilidades).
    n_resamples : int, default=2000

    Returns
    -------
    dict con ``point_estimate``, ``ci_low``, ``ci_high``, ``confidence_level``, ``method="BCa"``.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    rng = np.random.default_rng(random_state)

    point_estimate = float(metric_fn(y_true, y_score))

    boot_estimates = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_estimates[b] = metric_fn(y_true[idx], y_score[idx])

    # Corrección de sesgo z0
    proportion_less = np.mean(boot_estimates < point_estimate)
    proportion_less = np.clip(proportion_less, 1e-6, 1 - 1e-6)
    z0 = norm.ppf(proportion_less)

    # Aceleración a (jackknife, leave-one-out)
    jack_estimates = np.empty(n)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[:] = True
        mask[i] = False
        jack_estimates[i] = metric_fn(y_true[mask], y_score[mask])
    jack_mean = jack_estimates.mean()
    numerator = np.sum((jack_mean - jack_estimates) ** 3)
    denominator = 6.0 * (np.sum((jack_mean - jack_estimates) ** 2) ** 1.5)
    a = numerator / denominator if denominator != 0 else 0.0

    alpha = 1 - confidence_level
    z_lo, z_hi = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)

    def _adjusted_percentile(z: float) -> float:
        adjusted_z = z0 + (z0 + z) / (1 - a * (z0 + z))
        return float(norm.cdf(adjusted_z) * 100)

    p_lo, p_hi = _adjusted_percentile(z_lo), _adjusted_percentile(z_hi)
    p_lo, p_hi = np.clip([p_lo, p_hi], 0.0, 100.0)

    return {
        "point_estimate": point_estimate,
        "ci_low": float(np.percentile(boot_estimates, p_lo)),
        "ci_high": float(np.percentile(boot_estimates, p_hi)),
        "confidence_level": confidence_level,
        "method": "BCa",
        "n_resamples": n_resamples,
    }


# ---------------------------------------------------------------------------
# Orquestador — Etapa 1 -> Etapa 2 (condicional)
# ---------------------------------------------------------------------------


def run_hierarchical_comparison(
    results: ResultsDict,
    metric: str = "f1",
    alpha: float = 0.05,
    top_k_for_delong: int = 3,
    multiple_test_method: str = "holm",
) -> dict[str, Any]:
    """Ejecuta la comparación jerárquica completa: Friedman -> (si es
    significativo) Nemenyi + CD + DeLong sobre el top-k + Cliff's Delta.

    Returns
    -------
    dict con, como mínimo, ``friedman`` (resultado de ``friedman_test``).
    Si ``friedman["significant"]`` es False, se detiene ahí (documentado
    explícitamente en el resultado, ver ``stage_2_note``) — NO se ejecutan
    Nemenyi/CD/DeLong sobre datos donde no hay evidencia ómnibus de
    diferencia, tal como exige el curso.
    """
    performance_matrix = build_performance_matrix(results, metric=metric)
    friedman = friedman_test(performance_matrix, alpha=alpha)

    output: dict[str, Any] = {"metric": metric, "performance_matrix": performance_matrix, "friedman": friedman}

    if not friedman["significant"]:
        output["stage_2_note"] = (
            f"Friedman no significativo (p={friedman['p_value']:.4f} >= alpha={alpha}): no se reporta "
            "evidencia de superioridad de ninguna combinación. Dado que N_OUTER_FOLDS=3 (pocos bloques), "
            "esto puede reflejar falta de potencia estadística tanto como equivalencia real — ver "
            "limitación documentada en el docstring del módulo."
        )
        return output

    sig_matrix = nemenyi_posthoc(performance_matrix)
    cd = compute_critical_difference(
        k=friedman["k_combinations"], n_blocks=friedman["n_blocks"], alpha=alpha
    )
    output["nemenyi_p_matrix"] = sig_matrix
    output["critical_difference"] = cd

    top_labels = list(friedman["avg_ranks"].index[:top_k_for_delong])
    top_combos = [tuple(label.split("|")) for label in top_labels]
    delong_table = compare_top_combinations_delong(results, top_combos, multiple_test_method=multiple_test_method)

    cliffs_rows = []
    for combo_a, combo_b in combinations(top_combos, 2):
        values_a = performance_matrix[_combo_label(*combo_a)].to_numpy()
        values_b = performance_matrix[_combo_label(*combo_b)].to_numpy()
        delta = cliffs_delta(values_a, values_b)
        cliffs_rows.append(
            {
                "combo_a": _combo_label(*combo_a),
                "combo_b": _combo_label(*combo_b),
                "cliffs_delta": delta,
                "interpretation": interpret_cliffs_delta(delta),
            }
        )

    output["top_combos"] = top_labels
    output["delong_comparisons"] = delong_table
    output["cliffs_delta"] = pd.DataFrame(cliffs_rows)
    return output


def run_hierarchical_comparison_from_table(
    master_table: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    metric: str = "f1",
    alpha: float = 0.05,
    top_k_for_delong: int = 3,
    multiple_test_method: str = "holm",
    categorical_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Como ``run_hierarchical_comparison``, pero partiendo de la tabla
    maestra en CSV (``experiment_runner.load_master_table``) en vez de un
    dict de ``NestedCVResult`` en memoria — el caso real de
    ``05_results_analysis.ipynb``, corrido en una sesión NUEVA después de
    que ``04_run_experiments.ipynb`` ya guardó y cerró.

    Etapa 1 (Friedman/rangos) usa únicamente las columnas
    ``"{metric}_per_fold"`` de la tabla (barato). Etapa 2 (DeLong) SOLO
    reajusta el top-k (2-3 combinaciones) con sus ``best_params_per_fold``
    ya conocidos (ver ``evaluation.regenerate_outof_fold_predictions`` —
    un ``fit`` por fold, no una búsqueda de hiperparámetros), nunca las
    104 combinaciones completas.

    Parameters
    ----------
    master_table : pd.DataFrame
        Ver ``experiment_runner.load_master_table``.
    X, y : DataFrame, Series
        El MISMO dataset (mismas filas/orden) usado para generar
        ``master_table`` — necesario para que ``recover_outer_fold``
        reproduzca los folds correctos.

    Returns
    -------
    dict
        Misma forma que ``run_hierarchical_comparison``.
    """
    import json

    from src.models import MODEL_REGISTRY

    performance_matrix = build_performance_matrix_from_table(master_table, metric=metric)
    friedman = friedman_test(performance_matrix, alpha=alpha)

    output: dict[str, Any] = {"metric": metric, "performance_matrix": performance_matrix, "friedman": friedman}

    if not friedman["significant"]:
        output["stage_2_note"] = (
            f"Friedman no significativo (p={friedman['p_value']:.4f} >= alpha={alpha}): no se reporta "
            "evidencia de superioridad de ninguna combinación. Dado que N_OUTER_FOLDS=3 (pocos bloques), "
            "esto puede reflejar falta de potencia estadística tanto como equivalencia real — ver "
            "limitación documentada en el docstring del módulo."
        )
        return output

    sig_matrix = nemenyi_posthoc(performance_matrix)
    cd = compute_critical_difference(k=friedman["k_combinations"], n_blocks=friedman["n_blocks"], alpha=alpha)
    output["nemenyi_p_matrix"] = sig_matrix
    output["critical_difference"] = cd

    top_labels = list(friedman["avg_ranks"].index[:top_k_for_delong])
    top_combos = [tuple(label.split("|")) for label in top_labels]

    pooled: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for model_name, technique, method in top_combos:
        row = master_table[
            (master_table["model"] == model_name)
            & (master_table["technique"] == technique)
            & (master_table["method"] == method)
        ].iloc[0]
        best_params_per_fold = json.loads(row["best_params_per_fold"])
        pooled[_combo_label(model_name, technique, method)] = regenerate_outof_fold_predictions(
            MODEL_REGISTRY[model_name],
            technique,
            best_params_per_fold,
            X,
            y,
            n_outer_folds=int(row["n_outer_folds"]),
            random_state=int(row["random_state"]),
            categorical_columns=categorical_columns,
        )

    delong_table = delong_comparisons_from_pooled(pooled, multiple_test_method=multiple_test_method)

    cliffs_rows = []
    for combo_a, combo_b in combinations(top_combos, 2):
        values_a = performance_matrix[_combo_label(*combo_a)].to_numpy()
        values_b = performance_matrix[_combo_label(*combo_b)].to_numpy()
        delta = cliffs_delta(values_a, values_b)
        cliffs_rows.append(
            {
                "combo_a": _combo_label(*combo_a),
                "combo_b": _combo_label(*combo_b),
                "cliffs_delta": delta,
                "interpretation": interpret_cliffs_delta(delta),
            }
        )

    output["top_combos"] = top_labels
    output["delong_comparisons"] = delong_table
    output["cliffs_delta"] = pd.DataFrame(cliffs_rows)
    return output
