"""Corrida local 5 x 5 de los modelos que se evalúan en CPU.

Orden de costo: Naive Bayes, regresión logística, árbol de decisión y Random
Forest (XGBoost, SVM y KNN se corren en Colab, ver ``notebooks/12_colab_5x5.ipynb``
y ``notebooks/07_svm_gpu_colab.ipynb``). Reanudable: ``resume=True`` salta lo ya
guardado en ``results/experiments_5x5__local.csv``.

Uso (desde la raíz, entorno conda ``creditcard``)::

    python -u -m src.run_local_5x5
"""

from __future__ import annotations

import sys

from src import config
from src.data_loader import load_processed_dataset, split_features_target
from src.experiment_runner import run_experiments
from src.models import MODEL_REGISTRY
from src.preprocessing import remove_duplicate_rows

LOCAL_MODELS = ("naive_bayes", "logistic_regression", "decision_tree", "random_forest")
CHECKPOINT = config.RESULTS_DIR / "experiments_5x5__local.csv"

if __name__ == "__main__":
    df, _ = remove_duplicate_rows(load_processed_dataset())
    X, y = split_features_target(df)
    nj = config.OPTIMIZATION_N_JOBS
    run_experiments(
        X,
        y,
        models=[MODEL_REGISTRY[m] for m in (sys.argv[1:] or LOCAL_MODELS)],
        n_outer_folds=config.N_OUTER_FOLDS,
        n_inner_folds=config.N_INNER_FOLDS,
        optimizer_kwargs_by_method={
            "grid_search": {"n_jobs": nj},
            "random_search": {"n_iter": config.RANDOM_SEARCH_ITER, "n_jobs": nj},
            "bayesian_optuna": {"n_trials": config.OPTUNA_N_TRIALS, "n_jobs": nj},
            "genetic_deap": {
                "population_size": config.GA_POPULATION_SIZE,
                "n_generations": config.GA_N_GENERATIONS,
                "n_jobs": nj,
            },
        },
        checkpoint_path=CHECKPOINT,
        resume=True,
        n_jobs=1,
    )
