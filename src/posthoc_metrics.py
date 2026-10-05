"""Métricas complementarias calculadas A POSTERIORI sobre la tabla maestra.

El F1 de la clase minoritaria sigue siendo la métrica principal del
proyecto (``config.DEFAULT_SCORING``): es la que optimizan los cuatro
métodos de búsqueda y la que ordena los resultados. Este módulo agrega, solo
para reportar, métricas que NO dependen del umbral de decisión de 0.5
(ROC-AUC y PR-AUC / *average precision*) y la precisión/recall por fold, sin
modificar el esquema de ``results/experiments_master.csv`` (agregar columnas
a un CSV que se escribe de forma incremental rompería la reanudación).

Para cada fila de la tabla maestra se REAJUSTA el pipeline con los
``best_params`` ya guardados de cada fold externo (un ``fit`` por fold, no
una búsqueda — ver ``evaluation.regenerate_outof_fold_predictions``) y se
calculan las métricas por fold. Como control, se compara el F1 por fold
recalculado contra el de la tabla (``f1_check_max_abs_diff``): si difieren
de forma apreciable, el reajuste no reproduce el modelo evaluado y las
métricas nuevas de esa fila no son confiables.

No aplica a SVM: sus filas vienen de cuML en GPU (Colab) y reajustarlas con
``sklearn.svm.SVC`` daría otro modelo.

Uso (desde la raíz del proyecto, con el entorno conda ``creditcard``)::

    python -m src.posthoc_metrics
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src import config
from src.balancing import build_pipeline
from src.complexity_analysis import recover_outer_fold
from src.models import MODEL_REGISTRY
from src.nested_cv import compute_core_metrics
from src.preprocessing import get_preprocessing_steps

POSTHOC_METRICS_PATH = config.PROJECT_ROOT / "results" / "experiments_posthoc_metrics.csv"
POSTHOC_METRICS: tuple[str, ...] = ("f1", "precision", "recall", "roc_auc", "pr_auc")
COMBO_KEY = ["model", "technique", "method"]


def per_fold_metrics(
    model_name: str,
    technique: str,
    best_params_per_fold: list[dict],
    X: pd.DataFrame,
    y: pd.Series,
    n_outer_folds: int,
    random_state: int,
    categorical_columns: list[str] | None = None,
) -> dict[str, list[float]]:
    """Métricas por fold externo (incluye PR-AUC) reajustando con los
    hiperparámetros ya elegidos por el inner loop de cada fold."""
    spec = MODEL_REGISTRY[model_name]
    preprocessing_steps = get_preprocessing_steps(spec, categorical_columns=categorical_columns)
    out: dict[str, list[float]] = {m: [] for m in POSTHOC_METRICS}
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
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_core_metrics(y_test.to_numpy(), y_pred, y_proba)
        metrics["pr_auc"] = float(average_precision_score(y_test, y_proba))
        for name in POSTHOC_METRICS:
            out[name].append(metrics[name])
    return out


def compute_posthoc_table(
    master_table: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    categorical_columns: list[str] | None = None,
    path: Path = POSTHOC_METRICS_PATH,
    skip_models: tuple[str, ...] = ("svm",),
    resume: bool = True,
) -> pd.DataFrame:
    """Calcula (y guarda de forma incremental en ``path``) las métricas por
    fold de cada fila de ``master_table``, salvo ``skip_models``."""
    done: set[tuple[str, str, str]] = set()
    if resume and path.exists():
        prev = pd.read_csv(path)
        done = set(map(tuple, prev[COMBO_KEY].to_numpy()))

    for _, row in master_table.iterrows():
        key = (row["model"], row["technique"], row["method"])
        if row["model"] in skip_models or key in done:
            continue
        folds = per_fold_metrics(
            row["model"],
            row["technique"],
            json.loads(row["best_params_per_fold"]),
            X,
            y,
            n_outer_folds=int(row["n_outer_folds"]),
            random_state=int(row["random_state"]),
            categorical_columns=categorical_columns,
        )
        f1_table = np.array(json.loads(row["f1_per_fold"]))
        new = {
            "model": key[0],
            "technique": key[1],
            "method": key[2],
            "f1_check_max_abs_diff": float(np.max(np.abs(np.array(folds["f1"]) - f1_table))),
        }
        for name in POSTHOC_METRICS:
            new[f"{name}_posthoc_per_fold"] = json.dumps(folds[name])
            new[f"{name}_posthoc_mean"] = float(np.mean(folds[name]))
            new[f"{name}_posthoc_std"] = float(np.std(folds[name], ddof=1))
        pd.DataFrame([new]).to_csv(path, mode="a", header=not path.exists(), index=False)
        print(
            f"{key}: pr_auc={new['pr_auc_posthoc_mean']:.4f} f1_check={new['f1_check_max_abs_diff']:.2e}",
            flush=True,
        )
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


if __name__ == "__main__":
    from src.data_loader import load_processed_dataset, split_features_target
    from src.experiment_runner import load_master_table
    from src.preprocessing import remove_duplicate_rows

    df, _ = remove_duplicate_rows(load_processed_dataset())
    X_all, y_all = split_features_target(df)
    compute_posthoc_table(load_master_table(config.EXPERIMENTS_MASTER_PATH), X_all, y_all)
