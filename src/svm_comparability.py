"""Chequeo de comparabilidad de las 16 filas de SVM corridas en Colab (cuML).

Las filas de SVM de ``results/svm_gpu_results_colab.csv`` salieron de
``cuml.svm.SVC`` en GPU, no de ``sklearn.svm.SVC`` como los otros seis
modelos. Repetir la búsqueda completa en CPU no es viable (la corrida local
de ``svm/none/genetic_deap`` tardó más de 4 horas). Lo que sí se puede
verificar es la parte de la comparación que depende de la evaluación y no de
la búsqueda:

1. Se toman los ``best_params`` que eligió el inner loop de Colab en cada fold
   externo.
2. Se reajusta, con el MISMO pipeline del resto del proyecto
   (``balancing.build_pipeline``: preprocesamiento + balanceo + ``SVC`` de
   scikit-learn) sobre los MISMOS folds externos (``StratifiedKFold(3,
   shuffle=True, random_state=42)``), y se mide el F1 de prueba.
3. Se compara con el F1 por fold que reportó Colab.

Si el F1 recalculado coincide con el de Colab, las filas de SVM miden lo
mismo que las demás y son comparables en métrica. Lo que NO se puede
verificar con esto: que la búsqueda de cuML habría elegido los mismos
hiperparámetros que la de libsvm, ni los tiempos (GPU vs CPU).

Se ajusta con ``probability=False`` para no pagar el Platt scaling interno
(5 ajustes extra por fit): ``predict`` de ``SVC`` usa el signo de la función de
decisión con ambas opciones, así que F1, precisión y recall no cambian; el
ROC-AUC y el PR-AUC se calculan con ``decision_function``, que ordena igual
que cualquier probabilidad monótona.

Uso (desde la raíz, entorno conda ``creditcard``)::

    python -m src.svm_comparability
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config
from src.balancing import build_pipeline
from src.complexity_analysis import recover_outer_fold
from src.models import MODEL_REGISTRY
from src.preprocessing import get_preprocessing_steps

SVM_COLAB_PATH = config.RESULTS_DIR / "svm_gpu_results_colab.csv"
SVM_LOCAL_CHECK_PATH = config.RESULTS_DIR / "svm_local_refit_check.csv"
COMBO_KEY = ["model", "technique", "method"]
# El resumen del CSV de Colab se calculó sobre 3 folds; tolerancia con la que
# se considera que el F1 coincide (la diferencia esperada entre solvers es de
# orden 1e-3, mucho menor que la desviación entre folds, ~0.006).
F1_TOLERANCE = 0.01


def refit_one_combination(
    technique: str,
    best_params_per_fold: list[dict],
    X: pd.DataFrame,
    y: pd.Series,
    n_outer_folds: int,
    random_state: int,
) -> dict[str, list[float]]:
    """Reajusta con sklearn el SVM de cada fold externo con los ``best_params`` de Colab."""
    spec = MODEL_REGISTRY["svm"]
    steps = get_preprocessing_steps(spec)
    out: dict[str, list[float]] = {m: [] for m in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")}
    for fold_index, params in enumerate(best_params_per_fold):
        X_tr, X_te, y_tr, y_te = recover_outer_fold(
            X, y, fold_index, n_outer_folds=n_outer_folds, random_state=random_state
        )
        pipe = build_pipeline(
            spec, technique, y_train=y_tr, preprocessing_steps=steps, random_state=random_state, **params
        )
        pipe.set_params(model__probability=False, model__cache_size=300)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        score = pipe.decision_function(X_te)
        out["accuracy"].append(float(accuracy_score(y_te, y_pred)))
        out["precision"].append(float(precision_score(y_te, y_pred, zero_division=0)))
        out["recall"].append(float(recall_score(y_te, y_pred, zero_division=0)))
        out["f1"].append(float(f1_score(y_te, y_pred, zero_division=0)))
        out["roc_auc"].append(float(roc_auc_score(y_te, score)))
        out["pr_auc"].append(float(average_precision_score(y_te, score)))
    return out


def run_svm_comparability_check(
    X: pd.DataFrame,
    y: pd.Series,
    colab_path: Path = SVM_COLAB_PATH,
    out_path: Path = SVM_LOCAL_CHECK_PATH,
    reverse: bool = False,
) -> pd.DataFrame:
    """Reajusta las 16 filas de SVM en CPU y las compara con Colab (reanudable).

    ``reverse=True`` recorre las filas de atrás hacia adelante: sirve para
    lanzar un segundo proceso (con otro ``out_path``) que avance desde el
    otro extremo, ya que cada ajuste de libsvm usa un solo núcleo.
    """
    colab = pd.read_csv(colab_path)
    if reverse:
        colab = colab.iloc[::-1]
    done: set[tuple[str, str, str]] = set()
    for existing in {out_path, SVM_LOCAL_CHECK_PATH}:
        if existing.exists():
            done |= set(map(tuple, pd.read_csv(existing)[COMBO_KEY].to_numpy()))

    for _, row in colab.iterrows():
        key = (row["model"], row["technique"], row["method"])
        if key in done:
            continue
        t0 = time.perf_counter()
        folds = refit_one_combination(
            row["technique"],
            json.loads(row["best_params_per_fold"]),
            X,
            y,
            n_outer_folds=int(row["n_outer_folds"]),
            random_state=int(row["random_state"]),
        )
        f1_colab = np.array(json.loads(row["f1_per_fold"]))
        roc_colab = np.array(json.loads(row["roc_auc_per_fold"]))
        new = {
            "model": key[0],
            "technique": key[1],
            "method": key[2],
            "f1_colab_mean": float(f1_colab.mean()),
            "f1_local_mean": float(np.mean(folds["f1"])),
            "f1_max_abs_diff_per_fold": float(np.max(np.abs(np.array(folds["f1"]) - f1_colab))),
            "roc_auc_max_abs_diff_per_fold": float(np.max(np.abs(np.array(folds["roc_auc"]) - roc_colab))),
            "refit_seconds": time.perf_counter() - t0,
        }
        for name, values in folds.items():
            new[f"{name}_per_fold"] = json.dumps(values)
        pd.DataFrame([new]).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        print(
            f"{key}: f1 colab={new['f1_colab_mean']:.4f} local={new['f1_local_mean']:.4f} "
            f"max|dF1|={new['f1_max_abs_diff_per_fold']:.4f} ({new['refit_seconds']:.0f}s)",
            flush=True,
        )
    return pd.read_csv(out_path)


if __name__ == "__main__":
    from src.data_loader import load_processed_dataset, split_features_target
    from src.preprocessing import remove_duplicate_rows

    df, _ = remove_duplicate_rows(load_processed_dataset())
    X_all, y_all = split_features_target(df)
    reverse_run = "--reverse" in sys.argv
    run_svm_comparability_check(
        X_all,
        y_all,
        out_path=SVM_LOCAL_CHECK_PATH.with_name("svm_local_refit_check_reverse.csv") if reverse_run else SVM_LOCAL_CHECK_PATH,
        reverse=reverse_run,
    )
