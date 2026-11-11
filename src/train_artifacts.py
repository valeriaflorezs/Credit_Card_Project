"""Reentrena y serializa la mejor combinación de cada modelo benchmark (Entregable 3, sección 4.2).

Para cada uno de los 9 modelos de referencia se toma su mejor combinación de la validación anidada 5 × 5
(``experiments/comparacion_final/mejor_por_modelo.csv``: técnica de balanceo y método de búsqueda).

**Cómo se fijan los hiperparámetros finales.** El bucle interno eligió una configuración en cada uno de los
5 pliegues externos. Esas configuraciones (sin repetir) son los candidatos; cada una se evalúa con
validación cruzada estratificada de 5 pliegues sobre todos los datos (semilla 42, F1 de la clase de impago)
y se queda la de mayor F1 medio. Así no se abre un espacio de búsqueda nuevo: solo se elige entre lo que
ya eligió la validación anidada. Luego el pipeline completo (preprocesamiento + balanceo + modelo) se
reentrena con todos los datos y se guarda con ``joblib.dump(..., compress=3)``.

CS-FOF ya tiene su artefacto (``src/train_cs_fof_artifact.py``, búsqueda aleatoria sobre todos los datos) y
no se reentrena aquí; solo se agrega al índice.

Salidas en ``artifacts/``:
- ``<modelo>.joblib`` y ``metadata_<modelo>.json`` (versiones, semilla, hiperparámetros, esquema de
  entrada, métricas del bucle externo y resultado de la prueba de ida y vuelta);
- ``ida_y_vuelta.npz``: 200 clientes fijos y las probabilidades del modelo en memoria, que
  ``tests/test_artifacts.py`` compara con las del modelo cargado;
- ``indice.json``: lista de artefactos.

Uso (desde la raíz): python -m src.train_artifacts [modelo ...]
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
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import config
from src.balancing import build_pipeline
from src.data_loader import split_features_target
from src.models import MODEL_REGISTRY
from src.preprocessing import build_final_dataset, get_preprocessing_steps, load_raw_dataset, remove_duplicate_rows
from src.train_cs_fof_artifact import _esquema

ART = config.PROJECT_ROOT / "artifacts"
MEJORES = config.PROJECT_ROOT / "experiments" / "comparacion_final" / "mejor_por_modelo.csv"
MUESTRA = ART / "ida_y_vuelta.npz"
BENCHMARKS = ["catboost", "xgboost", "lightgbm", "random_forest", "svm", "knn", "decision_tree",
              "logistic_regression", "naive_bayes"]


def _metricas_externas(fila) -> dict:
    out = {}
    for m in ["f1", "precision", "recall", "roc_auc", "accuracy"]:
        col = f"{m}_per_fold"
        if col in fila and isinstance(fila[col], str):
            v = json.loads(fila[col])
            out[m] = {"media": float(np.mean(v)), "desviacion_estandar": float(np.std(v, ddof=1)), "por_pliegue": v}
    return out


def _pipeline(nombre, tecnica, y, params):
    spec = MODEL_REGISTRY[nombre]
    return build_pipeline(spec, tecnica, y_train=y, preprocessing_steps=get_preprocessing_steps(spec),
                          random_state=config.RANDOM_STATE, **params)


def entrenar(nombre, fila, X, y, muestra_X):
    t0 = time.time()
    tecnica, metodo = fila["technique"], fila["method"]
    candidatos = []
    for p in json.loads(fila["best_params_per_fold"]):
        if p not in candidatos:
            candidatos.append(p)
    cv = StratifiedKFold(config.N_INNER_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    puntajes = []
    for p in candidatos:
        s = cross_val_score(_pipeline(nombre, tecnica, y, p), X, y, scoring="f1", cv=cv, n_jobs=1)
        puntajes.append(float(s.mean()))
        print(f"  {nombre}: candidato {len(puntajes)}/{len(candidatos)} F1 {s.mean():.4f}", flush=True)
    k = int(np.argmax(puntajes)); params = candidatos[k]
    final = _pipeline(nombre, tecnica, y, params).fit(X, y)

    ruta = ART / f"{nombre}.joblib"
    joblib.dump(final, ruta, compress=3)
    p_memoria = final.predict_proba(muestra_X)[:, 1]
    p_cargado = joblib.load(ruta).predict_proba(muestra_X)[:, 1]
    ida_y_vuelta = bool(np.array_equal(p_memoria, p_cargado))

    meta = {
        "modelo": nombre, "archivo": ruta.name,
        "fecha_entrenamiento_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "semilla": config.RANDOM_STATE, "tecnica_balanceo": tecnica, "metodo_busqueda_anidada": metodo,
        "seleccion_final": {"criterio": "mayor F1 medio en CV estratificada de 5 pliegues sobre todos los datos, "
                                        "entre las configuraciones elegidas por el bucle interno",
                            "candidatos": candidatos, "f1_candidatos": puntajes, "elegido": k},
        "hiperparametros_finales": params,
        "metricas_bucle_externo_5x5": _metricas_externas(fila),
        "n_filas_entrenamiento": int(len(y)), "tasa_default": float(y.mean()),
        "columnas_entrada": _esquema(X),
        "clases": {"0": "no incumple", "1": "incumple el pago del mes siguiente"},
        "prueba_ida_y_vuelta": ida_y_vuelta,
        "tamano_archivo_mb": round(os.path.getsize(ruta) / 2**20, 2),
        "tiempo_entrenamiento_s": round(time.time() - t0, 1),
        "versiones": _versiones(),
    }
    (ART / f"metadata_{nombre}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{nombre}: F1 elegido {puntajes[k]:.4f}, ida y vuelta {ida_y_vuelta}, {meta['tamano_archivo_mb']} MB, "
          f"{meta['tiempo_entrenamiento_s']} s", flush=True)
    return p_memoria


def _versiones():
    v = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
         "scikit-learn": sklearn.__version__, "joblib": joblib.__version__}
    for lib in ("imblearn", "xgboost", "lightgbm", "catboost"):
        try:
            v[lib] = __import__(lib).__version__
        except ImportError:
            pass
    return v


def main(modelos):
    raw = load_raw_dataset()
    df, _ = build_final_dataset(raw)
    df, _ = remove_duplicate_rows(df)
    X, y = split_features_target(df)
    mejores = pd.read_csv(MEJORES).set_index("model")
    muestra_X = X.sample(200, random_state=config.RANDOM_STATE)

    previas = dict(np.load(MUESTRA, allow_pickle=False)) if MUESTRA.exists() else {}
    probs = {k: v for k, v in previas.items() if k.startswith("p_")}
    probs["p_cs_fof"] = joblib.load(ART / "cs_fof.joblib").predict_proba(muestra_X)[:, 1]
    for nombre in modelos:
        probs[f"p_{nombre}"] = entrenar(nombre, mejores.loc[nombre], X, y, muestra_X)
        np.savez(MUESTRA, X=muestra_X.to_numpy(float), columnas=np.array(list(X.columns)), **probs)

    indice = []
    for nombre in ["cs_fof"] + BENCHMARKS:
        mfile = ART / ("metadata.json" if nombre == "cs_fof" else f"metadata_{nombre}.json")
        if mfile.exists():
            m = json.loads(mfile.read_text(encoding="utf-8"))
            f1 = m.get("metricas_bucle_externo_5x5", {}).get("f1", {})
            indice.append({"modelo": nombre, "archivo": m["archivo"], "metadata": mfile.name,
                           "f1_externo_media": f1.get("media"), "f1_externo_de": f1.get("desviacion_estandar"),
                           "prueba_ida_y_vuelta": m.get("prueba_ida_y_vuelta"),
                           "tamano_archivo_mb": m.get("tamano_archivo_mb")})
    (ART / "indice.json").write_text(json.dumps(indice, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or BENCHMARKS)
