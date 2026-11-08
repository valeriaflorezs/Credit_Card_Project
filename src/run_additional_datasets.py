"""Evaluación de los 10 modelos en tres conjuntos de datos de crédito adicionales (Entregable 3, sección 6.3).

Conjuntos (repositorio UCI, descargados con ``ucimlrepo``):

- **German Credit** (Statlog, id 144): 1 000 clientes; clase positiva = mal pagador (30 %).
- **Australian Credit Approval** (Statlog, id 143): 690 solicitudes; clase positiva = rechazada (55 %).
- **Polish Companies Bankruptcy** (id 365), solo el año 1: ≈ 7 000 empresas, 64 razones financieras con
  faltantes; clase positiva = quiebra (≈ 4 %). Se usa un solo año para que la validación anidada de
  CS-FOF sea viable; se declara en ``experiments/protocolo.md``.

Procedimiento, igual para los 10 modelos y los tres conjuntos: validación cruzada anidada estratificada
5 x 5 (semilla 42), técnica ``class_weight`` y búsqueda aleatoria con 25 configuraciones sobre los
mismos espacios de ``src/models.py``, métrica F1 de la clase positiva. El preprocesamiento (imputación,
one-hot y escalado para los modelos sensibles a la escala) se ajusta dentro de cada pliegue.

Salida: ``results/additional_datasets/<conjunto>.csv`` (una fila por modelo y pliegue externo).
Reanudable: salta lo que ya está guardado.

Uso (desde la raíz)::

    python -u -m src.run_additional_datasets                       # los 3 conjuntos y los 10 modelos
    python -u -m src.run_additional_datasets german australian     # solo esos conjuntos
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.balancing import compute_class_weight_kwargs
from src.models import MODEL_REGISTRY, get_param_distributions

SALIDA = config.RESULTS_DIR / "additional_datasets"
CACHE = config.DATA_DIR / "external"
MODELOS = ("naive_bayes", "logistic_regression", "decision_tree", "knn", "svm",
           "random_forest", "xgboost", "lightgbm", "catboost", "cs_fof")

# Columnas categóricas codificadas como enteros en Australian (documentación de Statlog).
AUSTRALIAN_CATEGORICAS = ["A1", "A4", "A5", "A6", "A8", "A9", "A11", "A12"]


def cargar(nombre: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Devuelve X, y (1 = clase de riesgo) y la lista de columnas categóricas."""
    CACHE.mkdir(parents=True, exist_ok=True)
    ruta = CACHE / f"{nombre}.pkl"
    if ruta.exists():
        X, y_raw = pd.read_pickle(ruta)
    else:
        from ucimlrepo import fetch_ucirepo
        ids = {"german": 144, "australian": 143, "polish": 365}
        d = fetch_ucirepo(id=ids[nombre])
        X, y_raw = d.data.features.copy(), d.data.targets.iloc[:, 0].copy()
        pd.to_pickle((X, y_raw), ruta)
    if nombre == "german":
        y = (y_raw == 2).astype(int).to_numpy()            # 2 = mal pagador
        cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        X[cat] = X[cat].astype(object)
    elif nombre == "australian":
        y = (y_raw == 0).astype(int).to_numpy()            # 0 = solicitud rechazada
        cat = AUSTRALIAN_CATEGORICAS
    elif nombre == "polish":
        m = (X["year"] == 1).to_numpy()
        X, y = X.loc[m].drop(columns="year").reset_index(drop=True), y_raw[m].astype(int).to_numpy()
        cat = []
    else:
        raise ValueError(nombre)
    return X, y, cat


def pipeline(spec, X: pd.DataFrame, cat: list[str], y_train: np.ndarray) -> Pipeline:
    num = [c for c in X.columns if c not in cat]
    pasos_num = [("imputar", SimpleImputer(strategy="median"))]
    if spec.requires_scaling:
        pasos_num.append(("escalar", StandardScaler()))
    transformadores = [("num", Pipeline(pasos_num), num)]
    if cat:
        transformadores.append(("cat", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
        ]), cat))
    prep = ColumnTransformer(transformadores, sparse_threshold=0.0)
    modelo = spec.estimator_factory(random_state=config.RANDOM_STATE, **compute_class_weight_kwargs(spec, y_train))
    return Pipeline([("prep", prep), ("model", modelo)])


def evaluar(nombre: str, modelos: tuple[str, ...]) -> None:
    X, y, cat = cargar(nombre)
    SALIDA.mkdir(parents=True, exist_ok=True)
    archivo = SALIDA / f"{nombre}.csv"
    hechos = pd.read_csv(archivo) if archivo.exists() else pd.DataFrame(columns=["model", "fold"])
    externos = list(StratifiedKFold(config.N_OUTER_FOLDS, shuffle=True, random_state=config.RANDOM_STATE).split(X, y))
    print(f"\n== {nombre}: {X.shape[0]} filas, {X.shape[1]} variables, {y.mean():.1%} positivos", flush=True)
    for m in modelos:
        spec = MODEL_REGISTRY[m]
        n_jobs = 1 if m == "cs_fof" else config.OPTIMIZATION_N_JOBS   # CS-FOF paraleliza sus árboles
        dist = {f"model__{k}": v for k, v in get_param_distributions(spec).items()}
        for k, (tr, te) in enumerate(externos):
            if ((hechos["model"] == m) & (hechos["fold"] == k)).any():
                continue
            t0 = time.time()
            busqueda = RandomizedSearchCV(
                pipeline(spec, X, cat, y[tr]), dist, n_iter=config.RANDOM_SEARCH_ITER, scoring="f1",
                cv=StratifiedKFold(config.N_INNER_FOLDS, shuffle=True, random_state=config.RANDOM_STATE),
                random_state=config.RANDOM_STATE, n_jobs=n_jobs, refit=True, error_score=0.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                busqueda.fit(X.iloc[tr], y[tr])
            p = busqueda.predict_proba(X.iloc[te])[:, 1]
            yh = busqueda.predict(X.iloc[te])
            fila = dict(dataset=nombre, model=m, technique="class_weight", method="random_search", fold=k,
                        f1=f1_score(y[te], yh), precision=precision_score(y[te], yh, zero_division=0),
                        recall=recall_score(y[te], yh), roc_auc=roc_auc_score(y[te], p),
                        pr_auc=average_precision_score(y[te], p), inner_f1=busqueda.best_score_,
                        best_params=json.dumps({a.replace("model__", ""): (b.item() if hasattr(b, "item") else b)
                                                for a, b in busqueda.best_params_.items()}),
                        seconds=round(time.time() - t0, 1), random_state=config.RANDOM_STATE)
            hechos = pd.concat([hechos, pd.DataFrame([fila])], ignore_index=True)
            hechos.to_csv(archivo, index=False)
            print(f"  {m:20s} pliegue {k + 1}: F1 {fila['f1']:.4f}  AUC {fila['roc_auc']:.4f}  ({fila['seconds']:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    conjuntos = sys.argv[1:] or ["german", "australian", "polish"]
    for c in conjuntos:
        evaluar(c, MODELOS)
