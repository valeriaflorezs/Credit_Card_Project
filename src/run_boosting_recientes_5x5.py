"""Corrida 5 x 5 de los dos métodos recientes de boosting: LightGBM y CatBoost.

El Entregable 3 exige comparar el modelo original contra los modelos vistos en
clase y al menos dos métodos recientes del estado del arte. Se usa exactamente
el mismo protocolo que ``run_local_5x5.py``: validación anidada 5 x 5 con
semilla 42, las 4 técnicas de balanceo y los 4 métodos de búsqueda con los
mismos presupuestos (``config``).

Reanudable: ``resume=True`` salta lo ya guardado en el checkpoint.

Uso (desde la raíz, con el entorno del proyecto activado)::

    python -u -m src.run_boosting_recientes_5x5                 # ambos modelos
    python -u -m src.run_boosting_recientes_5x5 lightgbm        # solo uno
"""
from __future__ import annotations

import sys

from src import config
from src.data_loader import load_processed_dataset, split_features_target
from src.experiment_runner import run_experiments
from src.models import MODEL_REGISTRY
from src.preprocessing import remove_duplicate_rows

MODELOS = ("lightgbm", "catboost")
CHECKPOINT = config.RESULTS_DIR / "experiments_5x5__boosting_recientes.csv"

if __name__ == "__main__":
    df, _ = remove_duplicate_rows(load_processed_dataset())
    X, y = split_features_target(df)
    nj = config.OPTIMIZATION_N_JOBS
    run_experiments(
        X,
        y,
        models=[MODEL_REGISTRY[m] for m in (sys.argv[1:] or MODELOS)],
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
