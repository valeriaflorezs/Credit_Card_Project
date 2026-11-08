"""Entrena el modelo final de CS-FOF sobre TODO el dataset y lo serializa en artifacts/.

Sigue la Sección 4.2 del Entregable 3: con la combinación ganadora de la
validación anidada 5 x 5 (class_weight + búsqueda aleatoria), se repite esa
misma búsqueda (25 configuraciones, 5 pliegues internos, semilla 42, métrica F1)
sobre todos los datos, se reentrena el pipeline completo (codificación + modelo)
y se guarda con joblib junto a ``metadata.json``. El dashboard solo carga este
archivo: nunca reentrena.

Uso (desde la raíz del proyecto)::

    python -m src.train_cs_fof_artifact                      # usa data/raw/...xls
    python -m src.train_cs_fof_artifact ruta/al/archivo.xls
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src import config
from src.data_loader import split_features_target
from src.models import MODEL_REGISTRY, get_param_distributions
from src.original_model import CSFisherObliqueForest
from src.preprocessing import build_column_transformer, build_final_dataset, load_raw_dataset, remove_duplicate_rows

ARTIFACTS_DIR = config.PROJECT_ROOT / "artifacts"
MODELO_PATH = ARTIFACTS_DIR / "cs_fof.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
RESULTADOS_5X5 = config.RESULTS_DIR / "experiments_5x5__cs_fof.csv"


def _esquema(X: pd.DataFrame) -> list[dict]:
    """Nombre, tipo y rango válido de cada columna de entrada."""
    cols = []
    for c in X.columns:
        s = X[c]
        info = {"nombre": c, "tipo": "categorica" if c in config.CATEGORICAL_COLUMNS else "numerica"}
        if info["tipo"] == "categorica":
            info["valores"] = sorted(int(v) for v in s.unique())
        else:
            info["min"], info["max"] = float(s.min()), float(s.max())
        cols.append(info)
    return cols


if __name__ == "__main__":
    t0 = time.time()
    raw = load_raw_dataset(sys.argv[1] if len(sys.argv) > 1 else None)
    df, _ = build_final_dataset(raw)
    df, _ = remove_duplicate_rows(df)
    X, y = split_features_target(df)

    modelo = Pipeline([
        ("prep", build_column_transformer(apply_scaling=False)),   # one-hot, igual que en la validación
        ("model", CSFisherObliqueForest(class_weight="balanced", random_state=config.RANDOM_STATE,
                                        n_jobs=config.CSFOF_N_JOBS)),
    ])
    dist = {f"model__{k}": v for k, v in get_param_distributions(MODEL_REGISTRY["cs_fof"]).items()}
    busqueda = RandomizedSearchCV(
        modelo, dist, n_iter=config.RANDOM_SEARCH_ITER, scoring=config.DEFAULT_SCORING,
        cv=StratifiedKFold(config.N_INNER_FOLDS, shuffle=True, random_state=config.RANDOM_STATE),
        random_state=config.RANDOM_STATE, n_jobs=1, refit=True,
    )
    busqueda.fit(X, y)
    final = busqueda.best_estimator_

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODELO_PATH, compress=3)

    # Prueba de ida y vuelta: el modelo cargado debe predecir EXACTAMENTE lo mismo.
    cargado = joblib.load(MODELO_PATH)
    muestra = X.sample(2000, random_state=config.RANDOM_STATE)
    ida_y_vuelta = bool(np.array_equal(final.predict_proba(muestra), cargado.predict_proba(muestra)))

    metricas_5x5 = {}
    if RESULTADOS_5X5.exists():
        r = pd.read_csv(RESULTADOS_5X5)
        r = r[(r.technique == "class_weight") & (r.method == "random_search")]
        if len(r):
            for m in ["f1", "precision", "recall", "roc_auc", "accuracy"]:
                v = json.loads(r.iloc[0][f"{m}_per_fold"])
                metricas_5x5[m] = {"media": float(np.mean(v)), "desviacion_estandar": float(np.std(v, ddof=1)), "por_pliegue": v}

    params = {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v) for k, v in busqueda.best_params_.items()}
    metadata = {
        "modelo": "CS-FOF (Cost-Sensitive Fisher Oblique Forest)",
        "archivo": MODELO_PATH.name,
        "fecha_entrenamiento_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "semilla": config.RANDOM_STATE,
        "tecnica_balanceo": "class_weight (costo del moroso c = n0/n1)",
        "busqueda_final": {"metodo": "random_search", "configuraciones": config.RANDOM_SEARCH_ITER,
                            "pliegues_internos": config.N_INNER_FOLDS, "metrica": config.DEFAULT_SCORING,
                            "f1_validacion_interna": float(busqueda.best_score_)},
        "hiperparametros_finales": params,
        "fraccion_cortes_oblicuos": float(final.named_steps["model"].fraccion_cortes_oblicuos()),
        "metricas_bucle_externo_5x5": metricas_5x5,
        "n_filas_entrenamiento": int(len(y)), "tasa_default": float(y.mean()),
        "columnas_entrada": _esquema(X),
        "clases": {"0": "no incumple", "1": "incumple el pago del mes siguiente"},
        "prueba_ida_y_vuelta": ida_y_vuelta,
        "tamano_archivo_mb": round(os.path.getsize(MODELO_PATH) / 2**20, 2),
        "tiempo_entrenamiento_s": round(time.time() - t0, 1),
        "versiones": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                      "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ["hiperparametros_finales", "busqueda_final", "prueba_ida_y_vuelta",
                                                "tamano_archivo_mb", "tiempo_entrenamiento_s"]}, indent=2, ensure_ascii=False))
