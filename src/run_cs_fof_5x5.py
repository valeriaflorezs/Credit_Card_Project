"""Corrida 5 x 5 del modelo original CS-FOF (mismo protocolo que los 7 modelos base).

Usa la misma validación anidada (5 pliegues externos x 5 internos, semilla 42),
el mismo dataset procesado, las mismas 4 técnicas de balanceo y los mismos 4
métodos de búsqueda con los mismos presupuestos (``config``) que
``run_local_5x5.py``. Así la comparación con los modelos base es justa.

Única diferencia, solo de cómputo: el paralelismo. CS-FOF está escrito en
numpy y no libera el GIL de Python, así que los hilos que usan Optuna y el
algoritmo genético no lo aceleran. Por eso aquí la búsqueda evalúa un candidato
a la vez (``n_jobs=1``) y cada candidato reparte sus árboles en procesos
(``config.CSFOF_N_JOBS``). Los resultados no dependen de ese número.

Reanudable: ``resume=True`` salta lo ya guardado en el checkpoint.

Uso (desde la raíz, entorno conda ``creditcard``)::

    python -u -m src.run_cs_fof_5x5                      # las 16 combinaciones
    python -u -m src.run_cs_fof_5x5 class_weight none    # solo esas técnicas
"""
from __future__ import annotations

import sys

from src import config
from src.data_loader import load_processed_dataset, split_features_target
from src.experiment_runner import run_experiments
from src.models import MODEL_REGISTRY
from src.preprocessing import remove_duplicate_rows

CHECKPOINT = config.RESULTS_DIR / "experiments_5x5__cs_fof.csv"
TECNICAS = ("class_weight", "none", "smote", "adasyn")  # primero las más informativas

if __name__ == "__main__":
    df, _ = remove_duplicate_rows(load_processed_dataset())
    X, y = split_features_target(df)
    run_experiments(
        X,
        y,
        models=[MODEL_REGISTRY["cs_fof"]],
        techniques=list(sys.argv[1:] or TECNICAS),
        n_outer_folds=config.N_OUTER_FOLDS,
        n_inner_folds=config.N_INNER_FOLDS,
        optimizer_kwargs_by_method={
            "grid_search": {"n_jobs": 1},
            "random_search": {"n_iter": config.RANDOM_SEARCH_ITER, "n_jobs": 1},
            "bayesian_optuna": {"n_trials": config.OPTUNA_N_TRIALS, "n_jobs": 1},
            "genetic_deap": {
                "population_size": config.GA_POPULATION_SIZE,
                "n_generations": config.GA_N_GENERATIONS,
                "n_jobs": 1,
            },
        },
        checkpoint_path=CHECKPOINT,
        resume=True,
        n_jobs=1,
    )
