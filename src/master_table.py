"""Tabla maestra consolidada de las 112 combinaciones y selección de la mejor
por modelo.

Une en UN solo archivo versionado (``results/master_table_112.csv``) lo que
estaba repartido en tres: las 96 filas locales (``experiments_master.csv``),
las 16 de SVM (``svm_gpu_results_colab.csv``) y las métricas complementarias
(``experiments_posthoc_metrics.csv``). Cada fila lleva el origen del cálculo,
la semilla, el tiempo, F1 y complementarias como media y desviación estándar
entre los folds externos (``ddof=1``, la misma convención en todas las
columnas), el rango medio de F1 y si la combinación está dentro de la
diferencia crítica de Nemenyi respecto a la mejor global.

Uso (desde la raíz, entorno conda ``creditcard``)::

    python -m src.master_table
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.stats_comparison import compute_critical_difference

LOCAL_MASTER_PATH = config.EXPERIMENTS_MASTER_PATH
SVM_COLAB_PATH = config.SVM_COLAB_PATH
POSTHOC_PATH = config.RESULTS_DIR / "experiments_posthoc_metrics.csv"
SVM_LOCAL_CHECK_PATH = config.RESULTS_DIR / "svm_local_refit_check.csv"
MASTER_112_PATH = config.RESULTS_DIR / "master_table_112.csv"
BEST_PER_MODEL_PATH = config.RESULTS_DIR / "best_per_model.csv"

COMBO_KEY = ["model", "technique", "method"]
REPORTED_METRICS: tuple[str, ...] = ("f1", "precision", "recall", "roc_auc", "pr_auc", "accuracy")


def _per_fold(cell: str) -> list[float]:
    return json.loads(cell) if isinstance(cell, str) else []


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else np.nan


def load_svm_check(path: Path = SVM_LOCAL_CHECK_PATH) -> pd.DataFrame | None:
    """Reajuste local de las filas de SVM (``svm_comparability``), indexado por
    combinación. Une el archivo principal con el del segundo proceso
    (``*_reverse.csv``) si existe; ``None`` si no hay ninguno."""
    parts = [pd.read_csv(p) for p in (path, path.with_name(path.stem + "_reverse.csv")) if p.exists()]
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=COMBO_KEY).set_index(COMBO_KEY)


def build_master_table(
    local_path: Path = LOCAL_MASTER_PATH,
    svm_path: Path = SVM_COLAB_PATH,
    posthoc_path: Path = POSTHOC_PATH,
    svm_check_path: Path = SVM_LOCAL_CHECK_PATH,
) -> pd.DataFrame:
    """Arma la tabla de 112 filas, una por combinación (modelo, técnica, método).

    El PR-AUC y la precisión/recall de las 96 filas locales salen del
    reajuste posterior con los ``best_params`` guardados (ver
    ``posthoc_metrics``); las de SVM salen de ``svm_local_refit_check.csv``
    si existe (ver ``svm_comparability``) y quedan vacías si no.
    """
    local = pd.read_csv(local_path)
    local["source"] = "local_cpu_sklearn"
    svm = pd.read_csv(svm_path)
    svm["source"] = "colab_gpu_cuml"
    raw = pd.concat([local, svm], ignore_index=True)

    posthoc = pd.read_csv(posthoc_path).set_index(COMBO_KEY) if posthoc_path.exists() else None
    svm_check = load_svm_check(svm_check_path)

    rows = []
    for _, r in raw.iterrows():
        key = (r["model"], r["technique"], r["method"])
        row = {
            "model": key[0],
            "technique": key[1],
            "method": key[2],
            "source": r["source"],
            "random_state": int(r["random_state"]),
            "n_outer_folds": int(r["n_outer_folds"]),
            "n_inner_folds": int(r["n_inner_folds"]),
            "total_time_seconds": float(r["total_time_seconds"]),
            # El tiempo de SVM es de GPU: no se compara con el de las demás filas.
            "time_comparable": r["source"] == "local_cpu_sklearn",
        }
        per_fold = {m: _per_fold(r[f"{m}_per_fold"]) for m in ("accuracy", "f1", "precision", "recall", "roc_auc")}
        if posthoc is not None and key in posthoc.index:
            ph = posthoc.loc[key]
            per_fold["pr_auc"] = _per_fold(ph["pr_auc_posthoc_per_fold"])
            row["pr_auc_source"] = "reajuste posterior, predict_proba (posthoc_metrics)"
        elif svm_check is not None and key in svm_check.index:
            per_fold["pr_auc"] = _per_fold(svm_check.loc[key]["pr_auc_per_fold"])
            row["pr_auc_source"] = "reajuste en CPU, decision_function (svm_comparability)"
        else:
            per_fold["pr_auc"] = []
            row["pr_auc_source"] = ""
        for metric in REPORTED_METRICS:
            mean, std = _mean_std(per_fold[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_per_fold"] = json.dumps(per_fold[metric])
        # Solo SVM: F1 al reajustar con scikit-learn los best_params de Colab (ver svm_comparability).
        refit_f1 = _per_fold(svm_check.loc[key]["f1_per_fold"]) if (
            svm_check is not None and key in svm_check.index
        ) else []
        row["f1_cpu_refit_mean"], row["f1_cpu_refit_std"] = _mean_std(refit_f1)
        row["f1_cpu_refit_per_fold"] = json.dumps(refit_f1)
        row["best_params_per_fold"] = r["best_params_per_fold"]
        rows.append(row)

    table = pd.DataFrame(rows)

    # Rango medio de F1 entre las 112 combinaciones (1 = mejor), por fold externo.
    f1_matrix = np.array([_per_fold(c) for c in table["f1_per_fold"]])
    ranks = pd.DataFrame(f1_matrix).rank(axis=0, ascending=False, method="average")
    table["f1_avg_rank"] = ranks.mean(axis=1).to_numpy()
    table = table.sort_values("f1_avg_rank").reset_index(drop=True)

    cd = compute_critical_difference(k=len(table), n_blocks=f1_matrix.shape[1])
    table["within_nemenyi_cd_of_best"] = table["f1_avg_rank"] - table["f1_avg_rank"].iloc[0] < cd
    table.attrs["critical_difference"] = cd
    return table


def select_best_per_model(master: pd.DataFrame) -> pd.DataFrame:
    """Mejor combinación de cada modelo por F1 medio del bucle externo.

    Cada fila trae el criterio de selección escrito en claro: cuál es la
    ventaja sobre la segunda mejor del mismo modelo (en desviaciones
    estándar entre folds) y si la combinación entra en el conjunto de
    finalistas (a menos de la diferencia crítica de Nemenyi de la mejor
    global). Con 3 folds, que entre en ese conjunto no prueba equivalencia:
    refleja la poca potencia de la prueba.
    """
    cd = master.attrs.get("critical_difference", np.nan)
    global_best = master.sort_values("f1_mean", ascending=False).iloc[0]
    out = []
    for model, group in master.groupby("model"):
        ordered = group.sort_values("f1_mean", ascending=False)
        best, runner_up = ordered.iloc[0], ordered.iloc[1]
        gap = best["f1_mean"] - runner_up["f1_mean"]
        out.append(
            {
                "model": model,
                "technique": best["technique"],
                "method": best["method"],
                "f1_mean_std": f"{best['f1_mean']:.4f} ± {best['f1_std']:.4f}",
                "roc_auc_mean_std": f"{best['roc_auc_mean']:.4f} ± {best['roc_auc_std']:.4f}",
                "f1_mean": best["f1_mean"],
                "f1_std": best["f1_std"],
                "gap_to_runner_up": gap,
                "runner_up": f"{runner_up['technique']}|{runner_up['method']}",
                "within_nemenyi_cd_of_global_best": bool(best["within_nemenyi_cd_of_best"]),
                "is_global_best": bool(
                    (best["model"], best["technique"], best["method"])
                    == (global_best["model"], global_best["technique"], global_best["method"])
                ),
                "source": best["source"],
                "selection_reason": (
                    f"Mayor F1 medio externo entre las 16 combinaciones de {model}; ventaja de "
                    f"{gap:.4f} sobre {runner_up['technique']}|{runner_up['method']} "
                    f"({gap / best['f1_std']:.1f} desviaciones estándar entre folds)."
                    if best["f1_std"] and not np.isnan(best["f1_std"])
                    else f"Mayor F1 medio externo entre las 16 combinaciones de {model}."
                ),
            }
        )
    result = pd.DataFrame(out).sort_values("f1_mean", ascending=False).reset_index(drop=True)
    result.attrs["critical_difference"] = cd
    return result


def save_master_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera y guarda ``master_table_112.csv`` y ``best_per_model.csv``."""
    master = build_master_table()
    best = select_best_per_model(master)
    master.to_csv(MASTER_112_PATH, index=False)
    best.to_csv(BEST_PER_MODEL_PATH, index=False)
    return master, best


if __name__ == "__main__":
    master, best = save_master_tables()
    print(f"{len(master)} combinaciones -> {MASTER_112_PATH}")
    print(f"CD de Nemenyi (k={len(master)}, N=3): {master.attrs['critical_difference']:.2f}")
    print(best[["model", "technique", "method", "f1_mean_std", "within_nemenyi_cd_of_global_best"]].to_string())
