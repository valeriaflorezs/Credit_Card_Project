"""Corrida 5 x 5 de uno o más modelos dentro de una máquina de Colab.

Uso (en la VM, desde la carpeta del proyecto)::

    XGB_DEVICE=cuda python -u -m src.run_colab_5x5 xgboost

Guarda ``results/experiments_5x5__colab_<modelos>.csv`` (reanudable) y usa todos
los núcleos de la máquina para el paralelismo de cada método de optimización.
"""

from __future__ import annotations

import os
import sys

from src import config
from src.data_loader import load_processed_dataset, split_features_target
from src.experiment_runner import run_experiments
from src.models import MODEL_REGISTRY
from src.preprocessing import remove_duplicate_rows

if __name__ == "__main__":
    names = sys.argv[1:] or ["xgboost"]
    df, _ = remove_duplicate_rows(load_processed_dataset())
    assert df.shape == (29965, 21), df.shape
    X, y = split_features_target(df)
    nj = int(os.environ.get("N_JOBS", os.cpu_count() or 1))
    print(f"modelos={names} folds={config.N_OUTER_FOLDS}x{config.N_INNER_FOLDS} n_jobs={nj} "
          f"XGB_DEVICE={os.environ.get('XGB_DEVICE', 'cpu')}", flush=True)
    run_experiments(
        X, y,
        models=[MODEL_REGISTRY[m] for m in names],
        n_outer_folds=config.N_OUTER_FOLDS,
        n_inner_folds=config.N_INNER_FOLDS,
        optimizer_kwargs_by_method={
            "grid_search": {"n_jobs": nj},
            "random_search": {"n_iter": config.RANDOM_SEARCH_ITER, "n_jobs": nj},
            "bayesian_optuna": {"n_trials": config.OPTUNA_N_TRIALS, "n_jobs": nj},
            "genetic_deap": {"population_size": config.GA_POPULATION_SIZE,
                             "n_generations": config.GA_N_GENERATIONS, "n_jobs": nj},
        },
        checkpoint_path=config.RESULTS_DIR / f"experiments_5x5__colab_{'-'.join(names)}.csv",
        resume=True,
        n_jobs=1,
    )
