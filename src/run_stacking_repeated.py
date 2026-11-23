"""Stacking con validación cruzada repetida (R repeticiones x K pliegues externos).

Protocolo pre-registrado en ``experiments/protocolo_stacking.md``: hipótesis, contrastes y semillas
quedan fijados ahí ANTES de correr este script.

Para cada repetición r (semilla ``SEMILLA_BASE + r``) y cada pliegue externo k:

1. Modelos base (nivel 0): CS-FOF, XGBoost, CatBoost, LightGBM, Random Forest y regresión logística,
   todos con ponderación de clases y el mismo preprocesamiento del proyecto
   (``balancing.build_pipeline`` + ``preprocessing.get_preprocessing_steps``).
2. Con el entrenamiento externo se calculan las probabilidades *out-of-fold* (OOF) de cada modelo base
   con J pliegues internos. Después cada modelo base se reajusta con todo el entrenamiento externo y
   predice el pliegue de prueba.
3. Los combinadores de nivel 1 (``src/stacking.py``) se ajustan SOLO con las OOF, en dos versiones:
   ``_con`` (todos los modelos base) y ``_sin`` (sin CS-FOF). La diferencia mide el aporte de CS-FOF.
4. El umbral de cada candidato se elige con las OOF (máximo F1), nunca con la prueba. Para los
   combinadores que aprenden parámetros, las OOF de nivel 1 salen de otra validación cruzada interna.

Hiperparámetros de los modelos base:

- ``--busqueda 0`` (por defecto): los que guardó ``artifacts/metadata*.json``. Se eligieron con la
  búsqueda anidada sobre todos los datos, así que son algo optimistas, pero lo son igual para todos
  los modelos. Se declara como limitación en el protocolo.
- ``--busqueda N``: búsqueda aleatoria de N configuraciones (F1, J pliegues) dentro de cada
  entrenamiento externo. Es el protocolo estricto, pero tarda mucho más (sobre todo CS-FOF).

Salida (reanudable; salta los pares (r, k) ya guardados):

- ``<salida>/metricas.csv``: una fila por (repetición, pliegue, candidato).
- ``<salida>/predicciones/r<r>_k<k>.npz``: puntajes de prueba (no se versionan).

Uso (desde la raíz del repositorio)::

    python -u -m src.run_stacking_repeated                       # 10 x 5, configuración pre-registrada
    python -u -m src.run_stacking_repeated --repeticiones 2      # prueba corta
    python -u -m src.run_stacking_repeated --muestra 3000 --repeticiones 1 --salida results/stacking_prueba
    python -m src.stacking_analysis                              # pruebas estadísticas al terminar
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from src import config
from src.balancing import build_pipeline
from src.data_loader import load_processed_dataset, split_features_target
from src.models import MODEL_REGISTRY, get_param_distributions
from src.preprocessing import get_preprocessing_steps, remove_duplicate_rows
from src.stacking import Promedio, SeleccionCaruana, StackingLogistico, StackingPorSegmento, umbral_f1

# La regresión logística del proyecto usa ``penalty``, que scikit-learn 1.8+ marca como obsoleto; el
# aviso se repite en cada ajuste y no cambia el resultado.
warnings.filterwarnings("ignore", message=".*penalty.*", category=FutureWarning)
warnings.filterwarnings("ignore", message="Inconsistent values: penalty", category=UserWarning)

SEMILLA_BASE = 1000  # distinta de 42 (corrida principal) y de 1-4 (run_repeated_cv)
MODELOS_BASE = ("cs_fof", "xgboost", "catboost", "lightgbm", "random_forest", "logistic_regression")
MODELO_PROPIO = "cs_fof"
ARCHIVO_METADATA = {"cs_fof": "metadata.json"}  # los demás: metadata_<modelo>.json
COMBINADORES = {
    "stack_lr": StackingLogistico,
    "stack_fwls": StackingPorSegmento,
    "caruana": SeleccionCaruana,
    "promedio": Promedio,
}
SALIDA_POR_DEFECTO = config.RESULTS_DIR / "stacking_repeated"


def hiperparametros_artefacto(modelo: str) -> dict:
    archivo = ARCHIVO_METADATA.get(modelo, f"metadata_{modelo}.json")
    datos = json.loads((config.PROJECT_ROOT / "artifacts" / archivo).read_text(encoding="utf-8"))
    return dict(datos["hiperparametros_finales"])


def pipeline_base(modelo: str, y_tr: pd.Series, semilla: int, hp: dict):
    spec = MODEL_REGISTRY[modelo]
    return build_pipeline(spec, "class_weight", y_train=y_tr, preprocessing_steps=get_preprocessing_steps(spec),
                          random_state=semilla, **hp)


def buscar_hiperparametros(modelo, X_tr, y_tr, semilla, n_iter, cv, n_jobs) -> dict:
    spec = MODEL_REGISTRY[modelo]
    dist = {f"model__{k}": v for k, v in get_param_distributions(spec).items()}
    busqueda = RandomizedSearchCV(pipeline_base(modelo, y_tr, semilla, {}), dist, n_iter=n_iter, scoring="f1",
                                  cv=cv, random_state=semilla, n_jobs=1 if modelo == MODELO_PROPIO else n_jobs,
                                  refit=False)
    busqueda.fit(X_tr, y_tr)
    return {k.removeprefix("model__"): v for k, v in busqueda.best_params_.items()}


def metricas(y: np.ndarray, s: np.ndarray, umbral: float) -> dict:
    pred = (s >= umbral).astype(int)
    return {
        "umbral": umbral,
        "f1_umbral": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "f1_05": f1_score(y, (s >= 0.5).astype(int), zero_division=0),
        "roc_auc": roc_auc_score(y, s),
        "pr_auc": average_precision_score(y, s),
    }


def correr_pliegue(X_tr, y_tr, X_te, y_te, modelos, semilla, k, args) -> tuple[list[dict], dict]:
    cv_int = StratifiedKFold(args.internos, shuffle=True, random_state=semilla * 100 + k)
    oof, prueba, segundos, hps = {}, {}, {}, {}
    for m in modelos:
        t0 = time.perf_counter()
        hp = (buscar_hiperparametros(m, X_tr, y_tr, semilla, args.busqueda, cv_int, args.n_jobs)
              if args.busqueda else hiperparametros_artefacto(m))
        pipe = pipeline_base(m, y_tr, semilla, hp)
        nj = 1 if m == MODELO_PROPIO else args.n_jobs
        oof[m] = cross_val_predict(clone(pipe), X_tr, y_tr, cv=cv_int, method="predict_proba", n_jobs=nj)[:, 1]
        prueba[m] = pipe.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        segundos[m], hps[m] = time.perf_counter() - t0, hp
        print(f"    {m:<20} {segundos[m]:7.1f} s", flush=True)

    ytr, yte = y_tr.to_numpy(), y_te.to_numpy()
    g_tr = (X_tr["PAY_0"].to_numpy() > 0).astype(float)
    g_te = (X_te["PAY_0"].to_numpy() > 0).astype(float)
    filas, puntajes = [], {}

    for m in modelos:  # cada modelo base por separado, con su umbral elegido en las OOF
        puntajes[m] = prueba[m]
        filas.append({"candidato": m, "tipo": "base", **metricas(yte, prueba[m], umbral_f1(ytr, oof[m])),
                      "segundos": segundos[m], "pesos": json.dumps(hps[m], default=str)})

    variantes = {"con": list(modelos)}
    if MODELO_PROPIO in modelos and len(modelos) > 2:
        variantes["sin"] = [m for m in modelos if m != MODELO_PROPIO]
    cv_meta = StratifiedKFold(args.internos, shuffle=True, random_state=semilla * 100 + k + 50)
    for nombre, Clase in COMBINADORES.items():
        for var, miembros in variantes.items():
            t0 = time.perf_counter()
            P_tr = np.column_stack([oof[m] for m in miembros])
            P_te = np.column_stack([prueba[m] for m in miembros])
            # OOF de nivel 1 para elegir el umbral sin usar la prueba.
            s_meta = np.zeros(len(ytr))
            for a, b in cv_meta.split(P_tr, ytr):
                s_meta[b] = Clase().fit(P_tr[a], ytr[a], g_tr[a]).predict_proba(P_tr[b], g_tr[b])
            comb = Clase().fit(P_tr, ytr, g_tr)
            s_te = comb.predict_proba(P_te, g_te)
            cand = f"{nombre}_{var}"
            puntajes[cand] = s_te
            filas.append({"candidato": cand, "tipo": "ensamble", **metricas(yte, s_te, umbral_f1(ytr, s_meta)),
                          "segundos": time.perf_counter() - t0 + sum(segundos[m] for m in miembros),
                          "pesos": json.dumps(dict(zip(
                              miembros + ([f"{m}*atraso" for m in miembros] + ["atraso"] if nombre == "stack_fwls" else []),
                              comb.pesos())))})
    return filas, puntajes


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeticiones", type=int, default=10)
    ap.add_argument("--pliegues", type=int, default=config.N_OUTER_FOLDS)
    ap.add_argument("--internos", type=int, default=config.N_INNER_FOLDS)
    ap.add_argument("--modelos", nargs="+", default=list(MODELOS_BASE), choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--busqueda", type=int, default=0, help="0 = hiperparámetros de artifacts/; N = búsqueda aleatoria interna")
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("STACKING_N_JOBS", 4)),
                    help="procesos para los pliegues internos de los modelos rápidos (CS-FOF usa CSFOF_N_JOBS)")
    ap.add_argument("--muestra", type=int, default=0, help="submuestra estratificada para pruebas rápidas (0 = todo)")
    ap.add_argument("--salida", type=Path, default=SALIDA_POR_DEFECTO)
    args = ap.parse_args(argv)

    df, _ = remove_duplicate_rows(load_processed_dataset())
    X, y = split_features_target(df)
    if args.muestra:
        idx = StratifiedKFold(max(2, len(y) // args.muestra), shuffle=True, random_state=0).split(X, y)
        sel = next(idx)[1]
        X, y = X.iloc[sel].reset_index(drop=True), y.iloc[sel].reset_index(drop=True)
    args.salida.mkdir(parents=True, exist_ok=True)
    (args.salida / "predicciones").mkdir(exist_ok=True)
    ruta = args.salida / "metricas.csv"
    hechos = set()
    if ruta.exists():
        previo = pd.read_csv(ruta)
        hechos = set(zip(previo.repeticion, previo.pliegue))
    configuracion = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (args.salida / "configuracion.json").write_text(json.dumps(configuracion, indent=2), encoding="utf-8")
    print(f"n = {len(y)}, modelos = {args.modelos}, {args.repeticiones} x {args.pliegues} "
          f"(internos {args.internos}), búsqueda = {args.busqueda or 'artefactos'}", flush=True)

    for r in range(args.repeticiones):
        semilla = SEMILLA_BASE + r
        externos = StratifiedKFold(args.pliegues, shuffle=True, random_state=semilla)
        for k, (tr, te) in enumerate(externos.split(X, y)):
            if (r, k) in hechos:
                continue
            print(f"\n== repetición {r} (semilla {semilla}), pliegue {k}", flush=True)
            t0 = time.perf_counter()
            filas, puntajes = correr_pliegue(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te],
                                             args.modelos, semilla, k, args)
            tabla = pd.DataFrame(filas)
            tabla.insert(0, "repeticion", r)
            tabla.insert(1, "semilla", semilla)
            tabla.insert(2, "pliegue", k)
            tabla.insert(3, "n_train", len(tr))
            tabla.insert(4, "n_test", len(te))
            tabla.to_csv(ruta, mode="a", header=not ruta.exists(), index=False)
            np.savez_compressed(args.salida / "predicciones" / f"r{r}_k{k}.npz", indices=te,
                                y=y.iloc[te].to_numpy(), **{c: s.astype(np.float32) for c, s in puntajes.items()})
            mejor = tabla.sort_values("f1_umbral", ascending=False).head(4)
            print(f"   ({time.perf_counter() - t0:.0f} s) mejores F1: " +
                  ", ".join(f"{c} {v:.4f}" for c, v in zip(mejor.candidato, mejor.f1_umbral)), flush=True)
    print(f"\nListo: {ruta}. Siguiente paso: python -m src.stacking_analysis --salida {args.salida}")


if __name__ == "__main__":
    main()
