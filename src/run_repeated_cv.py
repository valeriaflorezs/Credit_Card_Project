"""Validación externa repetida (5 repeticiones x 5 pliegues) de las mejores combinaciones.

Con 5 pliegues externos las pruebas pareadas tienen poca potencia (Wilcoxon bilateral no puede
bajar de p = 0.0625). El Entregable 3 (sección 6.5) recomienda repetir la validación externa. Este
script repite la MISMA validación anidada 5 x 5 de ``run_experiments`` con otras semillas, solo para
la mejor combinación de cada uno de los tres mejores modelos:

- CS-FOF:        class_weight + búsqueda aleatoria
- XGBoost:       class_weight + bayesiana (Optuna)
- Random Forest: class_weight + búsqueda aleatoria

La semilla 42 ya está en la tabla maestra; aquí se corren las semillas 1, 2, 3 y 4. Cada semilla
cambia las particiones externas e internas y la semilla de los modelos. En total quedan
5 x 5 = 25 estimaciones pareadas por modelo.

Salida: ``results/repeated_cv/seed_<s>.csv``. Reanudable.

Uso (desde la raíz)::

    python -u -m src.run_repeated_cv            # semillas 1 a 4, los tres modelos
    python -u -m src.run_repeated_cv 1 2        # solo esas semillas
"""
from __future__ import annotations

import sys

from src import config
from src.data_loader import load_processed_dataset, split_features_target
from src.experiment_runner import run_experiments
from src.models import MODEL_REGISTRY
from src.preprocessing import remove_duplicate_rows

SEMILLAS = (1, 2, 3, 4)
COMBINACIONES = (  # (modelo, técnica, método): la mejor de cada modelo en la corrida con semilla 42
    ("xgboost", "class_weight", "bayesian_optuna"),
    ("random_forest", "class_weight", "random_search"),
    ("cs_fof", "class_weight", "random_search"),
)
SALIDA = config.RESULTS_DIR / "repeated_cv"


def _kwargs(modelo: str) -> dict:
    # CS-FOF es numpy puro: la búsqueda va de a un candidato y el modelo reparte sus árboles en procesos.
    nj = 1 if modelo == "cs_fof" else config.OPTIMIZATION_N_JOBS
    return {
        "random_search": {"n_iter": config.RANDOM_SEARCH_ITER, "n_jobs": nj},
        "bayesian_optuna": {"n_trials": config.OPTUNA_N_TRIALS, "n_jobs": nj},
    }


if __name__ == "__main__":
    semillas = [int(s) for s in sys.argv[1:]] or list(SEMILLAS)
    df, _ = remove_duplicate_rows(load_processed_dataset())
    X, y = split_features_target(df)
    SALIDA.mkdir(parents=True, exist_ok=True)
    for s in semillas:
        for modelo, tecnica, metodo in COMBINACIONES:
            print(f"\n== semilla {s}: {modelo} / {tecnica} / {metodo}", flush=True)
            run_experiments(
                X, y,
                models=[MODEL_REGISTRY[modelo]],
                techniques=[tecnica],
                methods=[metodo],
                n_outer_folds=config.N_OUTER_FOLDS,
                n_inner_folds=config.N_INNER_FOLDS,
                random_state=s,
                optimizer_kwargs_by_method=_kwargs(modelo),
                checkpoint_path=SALIDA / f"seed_{s}.csv",
                skip_log_path=SALIDA / f"seed_{s}_skipped.csv",
                resume=True,
                n_jobs=1,
            )
