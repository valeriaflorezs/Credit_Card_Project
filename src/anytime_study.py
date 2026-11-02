"""Estudio de curvas *anytime* y de multi-fidelidad.

La tabla maestra (``results/experiments_master.csv``) guarda solo el resultado
final de cada búsqueda; no dice cuánto tardó en encontrar un buen candidato.
Este estudio responde esa pregunta repitiendo, para un subconjunto de modelos
baratos, la optimización de las partes de entrenamiento de los mismos folds
externos del protocolo (``StratifiedKFold(5, shuffle=True, random_state=42)``)
con los presupuestos de ``config.py`` y registrando cada evaluación:

- ``grid_search``, ``random_search``, ``bayesian_optuna``, ``genetic_deap``:
  los cuatro métodos del proyecto, con su presupuesto de siempre.
- ``hyperband``: método multi-fidelidad propio (``src/multifidelity.py``),
  con un costo total comparable al de ``random_search``.

Por cada corrida se guarda el *trace* (score interno, costo y mejor score
hasta ese momento) y el desempeño del candidato final en el fold externo de
prueba. Todo se escribe por tarea en ``results/anytime/`` y es reanudable.

Los scores del trace son de la validación INTERNA (la que ve el optimizador);
el F1 del fold externo solo se usa para el resultado final, igual que en
``nested_cv.py``.

Uso (desde la raíz, entorno conda ``creditcard``)::

    python -m src.anytime_study --models logistic_regression decision_tree --folds 0 1 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import config
from src.models import MODEL_REGISTRY
from src.multifidelity import run_hyperband
from src.optimization import run_optimization
from src.preprocessing import get_preprocessing_steps, remove_duplicate_rows

ANYTIME_DIR = config.RESULTS_DIR / "anytime"
METHODS: tuple[str, ...] = (
    "grid_search",
    "random_search",
    "bayesian_optuna",
    "genetic_deap",
    "hyperband",
)
DEFAULT_TECHNIQUE = "class_weight"


def run_task(
    model_name: str,
    fold_index: int,
    seed: int,
    method: str,
    X: pd.DataFrame,
    y: pd.Series,
    technique: str = DEFAULT_TECHNIQUE,
    out_dir: Path = ANYTIME_DIR,
) -> None:
    """Corre un método sobre un fold externo y guarda trace + resultado final."""
    tag = f"{model_name}__{technique}__f{fold_index}__s{seed}__{method}"
    trace_path = out_dir / f"trace__{tag}.csv"
    final_path = out_dir / f"final__{tag}.csv"
    if trace_path.exists() and final_path.exists():
        return

    spec = MODEL_REGISTRY[model_name]
    outer_cv = StratifiedKFold(n_splits=config.N_OUTER_FOLDS, shuffle=True, random_state=seed)
    inner_cv = StratifiedKFold(n_splits=config.N_INNER_FOLDS, shuffle=True, random_state=seed)
    train_idx, test_idx = list(outer_cv.split(X, y))[fold_index]
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    steps = get_preprocessing_steps(spec)

    t0 = time.perf_counter()
    if method == "hyperband":
        result = run_hyperband(
            spec, technique, X_tr, y_tr, inner_cv, random_state=seed, preprocessing_steps=steps
        )
    else:
        result = run_optimization(
            method, spec, technique, X_tr, y_tr, inner_cv, random_state=seed, preprocessing_steps=steps
        )
    wall = time.perf_counter() - t0

    trace = result.diagnostics["trace"].copy()
    for col, value in (("model", model_name), ("technique", technique), ("fold", fold_index),
                       ("seed", seed), ("method", method)):
        trace.insert(0, col, value)
    if "fidelity" not in trace.columns:
        trace["fidelity"] = 1.0

    y_pred = result.best_pipeline.predict(X_te)
    y_proba = result.best_pipeline.predict_proba(X_te)[:, 1]
    final = pd.DataFrame([{
        "model": model_name, "technique": technique, "fold": fold_index, "seed": seed,
        "method": method, "n_evaluations": result.n_evaluations,
        "inner_best_score": result.best_score,
        "outer_f1": f1_score(y_te, y_pred, zero_division=0),
        "outer_roc_auc": roc_auc_score(y_te, y_proba),
        "total_cost_seconds": float(trace["cost_seconds"].sum()),
        "wall_seconds": wall,
        "full_fidelity_equivalents": result.diagnostics.get(
            "full_fidelity_equivalents", float(len(trace))
        ),
    }])
    out_dir.mkdir(parents=True, exist_ok=True)
    trace.to_csv(trace_path, index=False)
    final.to_csv(final_path, index=False)
    print(f"{tag}: inner={result.best_score:.4f} outer_f1={final['outer_f1'].iloc[0]:.4f} "
          f"cost={trace['cost_seconds'].sum():.0f}s", flush=True)


def load_anytime_results(out_dir: Path = ANYTIME_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatena los archivos por tarea en (traces, finals)."""
    traces = [pd.read_csv(p) for p in sorted(out_dir.glob("trace__*.csv"))]
    finals = [pd.read_csv(p) for p in sorted(out_dir.glob("final__*.csv"))]
    return (
        pd.concat(traces, ignore_index=True) if traces else pd.DataFrame(),
        pd.concat(finals, ignore_index=True) if finals else pd.DataFrame(),
    )


if __name__ == "__main__":
    from src.data_loader import load_processed_dataset, split_features_target

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "decision_tree", "xgboost"])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[config.RANDOM_STATE])
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--technique", default=DEFAULT_TECHNIQUE)
    args = parser.parse_args()

    df, _ = remove_duplicate_rows(load_processed_dataset())
    X_all, y_all = split_features_target(df)
    for model_name in args.models:
        for seed in args.seeds:
            for fold_index in args.folds:
                for method in args.methods:
                    run_task(model_name, fold_index, seed, method, X_all, y_all, args.technique)
