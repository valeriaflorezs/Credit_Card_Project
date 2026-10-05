"""Orquestador de experimentos: bucle parametrizado modelo × balanceo ×
método de optimización, paralelizado con joblib, con registro estructurado
en una tabla maestra (CSV) que se guarda de forma INCREMENTAL — cada
combinación agrega su fila apenas termina, no al final de todo el batch.

**112 combinaciones:** 7 modelos × 4 técnicas de balanceo × 4 métodos de
optimización. KNN y Naive Bayes no tienen ``class_weight`` nativo en
scikit-learn, pero ``models.BalancedKNeighborsClassifier`` y
``models.BalancedGaussianNB`` lo emulan (el de KNN es una extensión no
estándar), así que las 112 son ejecutables. En la corrida v2 SVM se
ejecuta aparte en Colab (16 combinaciones), por lo que esta tabla maestra
contiene 96.

**Registro incremental / reanudable:** cada combinación se agrega a
``results/experiments_master.csv`` apenas termina (no se espera a que
termine todo el batch), y ``run_experiments(resume=True)`` (default) salta
las combinaciones que ya aparecen en ese CSV. Esto permite correr un
subconjunto ahora (ej. modelos baratos en local), interrumpir, y correr el
resto después (ej. SVM/KNN en Colab) sin perder lo ya calculado — tal
como pide el proyecto para poder ejecutar subconjuntos de forma
independiente.

**Orden por costo computacional:** por defecto las combinaciones se
ordenan para que los modelos más baratos (Naive Bayes, Regresión
Logística, Decision Tree, XGBoost) corran primero — validan que el
pipeline funciona de punta a punta — dejando Random Forest, KNN y SVM
(los más costosos, sobre todo SVM+RBF con Grid Search) para el final.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as queue_module
import time
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

from src import config
from src.balancing import BALANCING_TECHNIQUES, is_compatible
from src.models import MODEL_REGISTRY, ModelSpec
from src.nested_cv import NestedCVResult, run_nested_cv
from src.optimization import OPTIMIZATION_METHODS

# Log separado (nunca en experiments_master.csv) para combinaciones
# canceladas por exceder ``timeout_seconds`` en ``run_experiments`` — ver
# docstring de esa funcion. Se excluyen de "pendientes" en corridas
# posteriores con resume=True, igual que las ya completadas, para no
# reintentarlas indefinidamente.
DEFAULT_SKIP_LOG_PATH = config.RESULTS_DIR / "skipped_timeout_log.csv"

# Heurística de costo computacional creciente (ver docstring del módulo).
# No es una medición exacta — solo ordena el trabajo pendiente para que lo
# barato corra primero.
MODEL_COST_ORDER: tuple[str, ...] = (
    "naive_bayes",
    "logistic_regression",
    "decision_tree",
    "xgboost",
    "random_forest",
    "knn",
    "svm",
)


def get_all_combinations(
    models: list[ModelSpec] | None = None,
    techniques: list[str] | None = None,
    methods: list[str] | None = None,
    sort_by_cost: bool = True,
) -> list[tuple[ModelSpec, str, str]]:
    """Construye la lista de combinaciones (modelo, técnica, método) VÁLIDAS.

    Filtra automáticamente las combinaciones (modelo, técnica) incompatibles
    (ver ``balancing.is_compatible``) — nunca se agenda una corrida
    condenada a fallar.

    Parameters
    ----------
    models : list[ModelSpec], optional
        Subconjunto de modelos a incluir. Por defecto, los 7 de
        ``models.MODEL_REGISTRY``.
    techniques : list[str], optional
        Subconjunto de técnicas de balanceo. Por defecto, las 4 de
        ``balancing.BALANCING_TECHNIQUES``.
    methods : list[str], optional
        Subconjunto de métodos de optimización. Por defecto, los 4 de
        ``optimization.OPTIMIZATION_METHODS``.
    sort_by_cost : bool, default=True
        Si True, ordena por ``MODEL_COST_ORDER`` (baratos primero).

    Returns
    -------
    list[tuple[ModelSpec, str, str]]
    """
    models = models if models is not None else list(MODEL_REGISTRY.values())
    techniques = techniques if techniques is not None else list(BALANCING_TECHNIQUES)
    methods = methods if methods is not None else list(OPTIMIZATION_METHODS)

    if sort_by_cost:
        order = {name: i for i, name in enumerate(MODEL_COST_ORDER)}
        models = sorted(models, key=lambda s: order.get(s.name, len(order)))

    combos: list[tuple[ModelSpec, str, str]] = []
    for spec in models:
        for technique in techniques:
            if not is_compatible(spec, technique):
                continue
            for method in methods:
                combos.append((spec, technique, method))
    return combos


def load_master_table(path: str | Path = config.EXPERIMENTS_MASTER_PATH) -> pd.DataFrame:
    """Carga la tabla maestra existente, o un DataFrame vacío si no existe todavía."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["model", "technique", "method"])
    return pd.read_csv(path)


def load_skip_log(path: str | Path = DEFAULT_SKIP_LOG_PATH) -> pd.DataFrame:
    """Carga el log de combinaciones saltadas por timeout, o vacío si no existe."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["model", "technique", "method"])
    return pd.read_csv(path)


def _pending_combinations(
    combos: list[tuple[ModelSpec, str, str]],
    master_table: pd.DataFrame,
    skip_log: pd.DataFrame | None = None,
) -> list[tuple[ModelSpec, str, str]]:
    done: set[tuple[str, str, str]] = set()
    if not master_table.empty:
        done |= set(zip(master_table["model"], master_table["technique"], master_table["method"]))
    if skip_log is not None and not skip_log.empty:
        done |= set(zip(skip_log["model"], skip_log["technique"], skip_log["method"]))
    if not done:
        return combos
    return [(s, t, m) for (s, t, m) in combos if (s.name, t, m) not in done]


def _append_row_to_csv(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    write_header = not path.exists()
    df_row.to_csv(path, mode="a", header=write_header, index=False)


def _build_row(
    result: NestedCVResult, random_state: int, n_inner_folds: int
) -> dict[str, Any]:
    # result.summary_row() ya incluye <metrica>_per_fold y
    # best_params_per_fold (ver nested_cv.py) — necesarios para que
    # evaluation.py/stats_comparison.py puedan trabajar desde esta tabla
    # en una sesión nueva, sin repetir la búsqueda de hiperparámetros.
    row = result.summary_row()
    row["random_state"] = random_state
    row["n_inner_folds"] = n_inner_folds
    return row


def run_single_experiment(
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
    """Corre UNA combinación (modelo, técnica, método) de punta a punta.

    Delgado wrapper sobre ``nested_cv.run_nested_cv`` — expuesto acá para
    poder probar/depurar una sola combinación sin pasar por el batch
    completo de ``run_experiments``.
    """
    return run_nested_cv(
        spec,
        technique,
        method,
        X,
        y,
        n_outer_folds=n_outer_folds,
        n_inner_folds=n_inner_folds,
        scoring=scoring,
        random_state=random_state,
        categorical_columns=categorical_columns,
        optimizer_kwargs=optimizer_kwargs,
    )


def _run_single_experiment_into_queue(
    q: mp.Queue,
    spec: ModelSpec,
    technique: str,
    method: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_outer_folds: int,
    n_inner_folds: int,
    scoring: str,
    random_state: int,
    categorical_columns: list[str] | None,
    optimizer_kwargs: dict[str, Any],
) -> None:
    """Target de ``multiprocessing.Process`` para poder aplicarle un timeout
    externo a una combinación (ver ``run_experiments(timeout_seconds=...)``).

    Debe ser una función de módulo (no un closure/lambda) para que
    ``multiprocessing`` (spawn, default en Windows) pueda importarla en el
    proceso hijo.
    """
    try:
        result = run_single_experiment(
            spec,
            technique,
            method,
            X,
            y,
            n_outer_folds=n_outer_folds,
            n_inner_folds=n_inner_folds,
            scoring=scoring,
            random_state=random_state,
            categorical_columns=categorical_columns,
            optimizer_kwargs=optimizer_kwargs,
        )
        q.put(("ok", result))
    except Exception as exc:  # noqa: BLE001 — se reporta tal cual al padre
        q.put(("error", repr(exc)))


def run_experiments(
    X: pd.DataFrame,
    y: pd.Series,
    models: list[ModelSpec] | None = None,
    techniques: list[str] | None = None,
    methods: list[str] | None = None,
    n_outer_folds: int = config.N_OUTER_FOLDS,
    n_inner_folds: int = config.N_INNER_FOLDS,
    scoring: str = config.DEFAULT_SCORING,
    random_state: int = config.RANDOM_STATE,
    categorical_columns: list[str] | None = None,
    optimizer_kwargs_by_method: dict[str, dict[str, Any]] | None = None,
    n_jobs: int = 1,
    backend: str = "loky",
    checkpoint_path: str | Path = config.EXPERIMENTS_MASTER_PATH,
    resume: bool = True,
    timeout_seconds: float | None = None,
    skip_log_path: str | Path = DEFAULT_SKIP_LOG_PATH,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], NestedCVResult]]:
    """Corre el batch completo (o un subconjunto) de experimentos.

    Parameters
    ----------
    X, y : DataFrame, Series
        Dataset COMPLETO ya deduplicado (ver
        ``preprocessing.remove_duplicate_rows``, aplicar UNA vez antes de
        llamar esta función, nunca dentro del bucle).
    models, techniques, methods : list, optional
        Subconjuntos a correr (ver ``get_all_combinations``) — permite
        correr, por ejemplo, solo los modelos baratos ahora y el resto
        (SVM/KNN) después, posiblemente en otra máquina (Colab).
    n_outer_folds, n_inner_folds : int
        Ver ``nested_cv.run_nested_cv``.
    scoring : str, default=config.DEFAULT_SCORING
    random_state : int, default=config.RANDOM_STATE
    categorical_columns : list[str], optional
        Ver ``preprocessing.get_preprocessing_steps``.
    optimizer_kwargs_by_method : dict, optional
        Overrides por método, ej. ``{"genetic_deap": {"population_size": 10,
        "n_generations": 6}}`` para reducir el costo en modelos caros.
    n_jobs : int, default=1
        Núcleos para ``joblib.Parallel``. 1 = secuencial (default seguro
        dado el hardware CPU-only de 16GB RAM del proyecto). Aumentar con
        cautela: cada worker de ``loky`` recibe su propia copia
        serializada de ``X``/``y``, lo que añade overhead de memoria por
        worker.
    backend : {"loky", "threading"}, default="loky"
        ``"loky"`` (procesos): evita la contención del GIL de Python en
        las partes puras de Optuna/DEAP, pero serializa X/y por worker.
        ``"threading"``: sin overhead de serialización, pero solo ayuda en
        las partes que ya liberan el GIL (la mayoría del cómputo numérico
        de scikit-learn/XGBoost), así que escala peor con estos
        optimizadores manuales.
    checkpoint_path : str or Path, default=config.EXPERIMENTS_MASTER_PATH
        CSV donde se registra cada corrida, de forma incremental.
    resume : bool, default=True
        Si True, salta combinaciones que ya están en ``checkpoint_path``
        (comparando modelo+técnica+método) en vez de recalcularlas. También
        salta las que ya estén en ``skip_log_path`` (ver ``timeout_seconds``).
    timeout_seconds : float, optional, default=None
        Si se da, cada combinación corre en un proceso aparte
        (``multiprocessing.Process``) con este tope de tiempo de pared. Si lo
        excede, el proceso se termina, la combinación se registra en
        ``skip_log_path`` (NO en ``checkpoint_path`` — nunca se mezclan filas
        sin métricas reales con la tabla maestra) y se sigue con la
        siguiente. Documentado como desviación explícita, no como omisión
        silenciosa — ver ``EXPERIMENT_STATUS.md``. Con ``None`` (default) el
        comportamiento es el original: ``joblib.Parallel`` sin timeout.
    skip_log_path : str or Path, default=DEFAULT_SKIP_LOG_PATH
        CSV donde se registran las combinaciones saltadas por timeout
        (columnas: model, technique, method, elapsed_seconds,
        timeout_seconds, reason). Se excluyen de "pendientes" en corridas
        futuras con ``resume=True``, igual que las completadas — para
        reintentarlas explícitamente, pasar ``models``/``techniques``/
        ``methods`` acotado a esas combinaciones con un ``timeout_seconds``
        mayor (o None).

    Returns
    -------
    master_table : pd.DataFrame
        Tabla completa (corridas previas + nuevas), una fila por combinación.
    results : dict[(model, technique, method), NestedCVResult]
        SOLO las combinaciones corridas en ESTA llamada (para inspección
        en memoria, ej. ``fold_results`` para evaluation.py/interpretability.py).
    """
    checkpoint_path = Path(checkpoint_path)
    skip_log_path = Path(skip_log_path)
    combos = get_all_combinations(models, techniques, methods)

    existing_table = load_master_table(checkpoint_path) if resume else pd.DataFrame()
    skip_log = load_skip_log(skip_log_path) if resume else pd.DataFrame()
    pending = _pending_combinations(combos, existing_table, skip_log) if resume else combos

    n_skipped = len(combos) - len(pending)
    if n_skipped > 0:
        print(
            f"[experiment_runner] {n_skipped} combinaciones ya en {checkpoint_path} o {skip_log_path}, "
            "se omiten (resume=True)."
        )
    print(f"[experiment_runner] {len(pending)} combinaciones a ejecutar.")

    if not pending:
        return existing_table, {}

    new_rows: list[dict[str, Any]] = []
    results: dict[tuple[str, str, str], NestedCVResult] = {}

    if timeout_seconds is not None:
        for spec, technique, method in pending:
            opt_kwargs = (optimizer_kwargs_by_method or {}).get(method, {})
            q: mp.Queue = mp.Queue()
            proc = mp.Process(
                target=_run_single_experiment_into_queue,
                args=(
                    q,
                    spec,
                    technique,
                    method,
                    X,
                    y,
                    n_outer_folds,
                    n_inner_folds,
                    scoring,
                    random_state,
                    categorical_columns,
                    opt_kwargs,
                ),
            )
            start = time.time()
            proc.start()
            # Se vacía la cola ANTES de hacer join: el resultado (con y_true/y_pred/
            # y_proba por fold) puede superar el buffer de la tubería, y un hijo
            # bloqueado en q.put() nunca termina, lo que se confundiría con un timeout.
            try:
                status, payload = q.get(timeout=timeout_seconds)
                reason = None if status == "ok" else "error"
                proc.join()
            except queue_module.Empty:
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
                    reason = "timeout"
                else:
                    reason = "process_died"
                status, payload = None, None
            elapsed = time.time() - start

            if reason is not None or status != "ok":
                detail = f"tras {elapsed:.1f}s" + (f": {payload}" if payload else "")
                print(f"[experiment_runner] SKIP ({reason}) {spec.name}/{technique}/{method} {detail}")
                _append_row_to_csv(
                    {
                        "model": spec.name,
                        "technique": technique,
                        "method": method,
                        "elapsed_seconds": elapsed,
                        "timeout_seconds": timeout_seconds,
                        "reason": reason,
                        "detail": payload,
                    },
                    skip_log_path,
                )
                continue

            result = payload
            row = _build_row(result, random_state=random_state, n_inner_folds=n_inner_folds)
            _append_row_to_csv(row, checkpoint_path)
            new_rows.append(row)
            results[(spec.name, technique, method)] = result
            print(
                f"[experiment_runner] OK {spec.name}/{technique}/{method} — "
                f"f1={result.metrics_summary['f1'][0]:.4f}±{result.metrics_summary['f1'][1]:.4f} "
                f"({result.total_time_seconds:.1f}s)"
            )
    else:

        def _run_one(spec: ModelSpec, technique: str, method: str):
            opt_kwargs = (optimizer_kwargs_by_method or {}).get(method, {})
            result = run_single_experiment(
                spec,
                technique,
                method,
                X,
                y,
                n_outer_folds=n_outer_folds,
                n_inner_folds=n_inner_folds,
                scoring=scoring,
                random_state=random_state,
                categorical_columns=categorical_columns,
                optimizer_kwargs=opt_kwargs,
            )
            return spec.name, technique, method, result

        parallel = Parallel(n_jobs=n_jobs, backend=backend, return_as="generator")

        for model_name, technique, method, result in parallel(
            delayed(_run_one)(spec, technique, method) for spec, technique, method in pending
        ):
            row = _build_row(result, random_state=random_state, n_inner_folds=n_inner_folds)
            _append_row_to_csv(row, checkpoint_path)
            new_rows.append(row)
            results[(model_name, technique, method)] = result
            print(
                f"[experiment_runner] OK {model_name}/{technique}/{method} — "
                f"f1={result.metrics_summary['f1'][0]:.4f}±{result.metrics_summary['f1'][1]:.4f} "
                f"({result.total_time_seconds:.1f}s)"
            )

    new_table = pd.DataFrame(new_rows)
    master_table = (
        pd.concat([existing_table, new_table], ignore_index=True) if not existing_table.empty else new_table
    )
    return master_table, results
