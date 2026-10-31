"""Une los CSV por sesión de ``10_colab_5x5.ipynb`` en la tabla maestra 5 x 5.

Cada sesión de Colab guarda ``experiments_5x5__<modelos>__<técnicas>.csv``. Se
copian esos archivos a ``results/`` (o a ``results/colab_5x5/``) y este script
los concatena en ``config.EXPERIMENTS_MASTER_PATH``, verificando que todas las
filas sean 5 x 5 con la misma semilla, que no haya combinaciones repetidas y
cuántas faltan de las 96 esperadas (6 modelos sin SVM; SVM va por cuML).

Uso (desde la raíz, entorno conda ``creditcard``)::

    python -m src.merge_session_results
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.experiment_runner import get_all_combinations
from src.models import MODEL_REGISTRY

SESSION_GLOB = "experiments_5x5__*.csv"


def merge_session_results() -> pd.DataFrame:
    files = sorted(config.RESULTS_DIR.rglob(SESSION_GLOB))
    if not files:
        raise FileNotFoundError(f"No hay archivos {SESSION_GLOB} bajo {config.RESULTS_DIR}")
    table = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    key = ["model", "technique", "method"]
    duplicated = table.duplicated(key, keep="last")
    if duplicated.any():
        print(f"Aviso: {int(duplicated.sum())} combinaciones repetidas entre sesiones; se conserva la última.")
        table = table.drop_duplicates(key, keep="last")
    assert (table["n_outer_folds"] == config.N_OUTER_FOLDS).all(), "hay filas que no son de 5 folds externos"
    assert (table["n_inner_folds"] == config.N_INNER_FOLDS).all(), "hay filas que no son de 5 folds internos"
    assert table["random_state"].nunique() == 1, "semillas distintas entre sesiones"

    expected = {
        (spec.name, t, m)
        for spec, t, m in get_all_combinations([s for n, s in MODEL_REGISTRY.items() if n != "svm"])
    }
    missing = sorted(expected - set(map(tuple, table[key].to_numpy())))
    print(f"{len(table)} filas de {len(files)} archivos; faltan {len(missing)} de las {len(expected)} esperadas (sin SVM).")
    for combo in missing[:20]:
        print("  falta:", combo)
    table.to_csv(config.EXPERIMENTS_MASTER_PATH, index=False)
    print("Guardado en", config.EXPERIMENTS_MASTER_PATH)
    return table


if __name__ == "__main__":
    merge_session_results()
