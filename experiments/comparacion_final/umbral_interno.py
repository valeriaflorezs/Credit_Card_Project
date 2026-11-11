"""Umbral de decisión elegido DENTRO del entrenamiento de cada pliegue externo (análisis exploratorio).

Pregunta: ¿cambia la comparación si cada modelo usa el umbral que maximiza su F1 en lugar de 0.5?

Protocolo, igual para los 5 mejores modelos (CS-FOF, CatBoost, XGBoost, LightGBM y Random Forest):
1. En cada pliegue externo k se toma SOLO su entrenamiento. Con los hiperparámetros que el bucle interno
   eligió en ese pliegue, se generan predicciones fuera de pliegue internas (5 pliegues estratificados,
   semilla 42).
2. Con esas predicciones internas se elige el umbral que maximiza el F1 (rejilla de 0.05 a 0.95, paso 0.01).
3. El umbral queda congelado y se aplica a las probabilidades del modelo entrenado con todo el entrenamiento
   del pliegue (``probabilidades_oof.npz``) sobre la prueba externa, que no intervino en ninguna elección.

Se reporta por separado el F1 con umbral 0.5 y con umbral ajustado. Pruebas sobre el F1 ajustado por
pliegue: t corregida de Nadeau–Bengio y Wilcoxon con Holm, más bootstrap pareado del F1 global.
Es exploratorio: se decidió después de ver los resultados y no sustenta por sí solo la condición R3.

Uso (desde la raíz): python experiments/comparacion_final/umbral_interno.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist, wilcoxon
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ / "experiments" / "robustez"))
from ampliada import _datos, _params  # noqa: E402
from robustez import fabrica  # noqa: E402

MODELOS = ["cs_fof", "catboost", "xgboost", "lightgbm", "random_forest"]
GRILLA = np.round(np.arange(0.05, 0.951, 0.01), 2)


def mejor_umbral(y, p):
    f = [f1_score(y, p >= u, zero_division=0) for u in GRILLA]
    return float(GRILLA[int(np.argmax(f))])


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); prev = 0
    for j, i in enumerate(o):
        prev = max(prev, min(1, (len(p) - j) * p[i])); adj[i] = prev
    return adj


def main():
    X, y, _, folds = _datos()
    P = _params()
    oof = np.load(AQUI / "probabilidades_oof.npz")
    salida = AQUI / "umbral_interno_pliegues.csv"
    hechos = pd.read_csv(salida) if salida.exists() else pd.DataFrame(columns=["model", "fold"])
    for m in MODELOS:
        for k, (tr, te) in enumerate(folds):
            if ((hechos["model"] == m) & (hechos["fold"] == k)).any():
                continue
            p_int = np.zeros(len(tr))
            for itr, iva in StratifiedKFold(5, shuffle=True, random_state=42).split(X[tr], y[tr]):
                mod = fabrica(m, P[m][k], y[tr][itr]).fit(X[tr][itr], y[tr][itr])
                p_int[iva] = mod.predict_proba(X[tr][iva])[:, 1]
            u = mejor_umbral(y[tr], p_int)
            p_te = oof[m][te]
            fila = dict(model=m, fold=k, umbral=u, f1_interno_05=f1_score(y[tr], p_int >= 0.5),
                        f1_interno_ajustado=f1_score(y[tr], p_int >= u),
                        f1_05=f1_score(y[te], p_te >= 0.5), f1_ajustado=f1_score(y[te], p_te >= u),
                        recall_05=float((p_te[y[te] == 1] >= 0.5).mean()), recall_ajustado=float((p_te[y[te] == 1] >= u).mean()))
            hechos = pd.concat([hechos, pd.DataFrame([fila])], ignore_index=True); hechos.to_csv(salida, index=False)
            print(f"{m:14s} pliegue {k + 1}: umbral {u:.2f}  F1 0.5 {fila['f1_05']:.4f} -> ajustado {fila['f1_ajustado']:.4f}", flush=True)

    H = pd.read_csv(salida)
    R = H.groupby("model").agg(umbral_medio=("umbral", "mean"), umbral_min=("umbral", "min"), umbral_max=("umbral", "max"),
                               f1_05=("f1_05", "mean"), f1_05_de=("f1_05", "std"),
                               f1_ajustado=("f1_ajustado", "mean"), f1_ajustado_de=("f1_ajustado", "std"))
    R["ganancia"] = R.f1_ajustado - R.f1_05
    R = R.reindex(MODELOS)
    R.to_csv(AQUI / "umbral_interno_resumen.csv")
    print(R.round(4).to_string())

    # pruebas sobre el F1 ajustado por pliegue: CS-FOF frente a cada rival
    F = {m: H[H.model == m].sort_values("fold").f1_ajustado.to_numpy() for m in MODELOS}
    n = 5; filas = []
    for r in MODELOS[1:]:
        d = F["cs_fof"] - F[r]
        tt = d.mean() / np.sqrt((1 / n + 1 / 4) * d.var(ddof=1))
        filas.append(dict(rival=r, dif_f1=d.mean(), gana=int((d > 0).sum()),
                          p_nadeau_bengio=2 * tdist.sf(abs(tt), n - 1), p_wilcoxon=wilcoxon(F["cs_fof"], F[r]).pvalue))
    T = pd.DataFrame(filas)
    T["p_nb_holm"] = holm(T.p_nadeau_bengio); T["p_w_holm"] = holm(T.p_wilcoxon)

    # bootstrap pareado del F1 global con el umbral congelado de cada pliegue
    yy = oof["y"].astype(int)
    pred = {}
    for m in MODELOS:
        pm = np.zeros(len(yy), bool)
        for k, (_, te) in enumerate(folds):
            u = float(H[(H.model == m) & (H.fold == k)].umbral.iloc[0])
            pm[te] = oof[m][te] >= u
        pred[m] = pm
    rng = np.random.default_rng(42); idx = {c: np.flatnonzero(yy == c) for c in (0, 1)}

    def f1w(p, w):
        tp = (w * (p & (yy == 1))).sum(); fp = (w * (p & (yy == 0))).sum(); fn = (w * (~p & (yy == 1))).sum()
        return 2 * tp / (2 * tp + fp + fn)

    reps = {r: [] for r in MODELOS[1:]}
    for _ in range(5000):
        w = sum(np.bincount(rng.choice(idx[c], len(idx[c]), replace=True), minlength=len(yy)) for c in (0, 1))
        for r in MODELOS[1:]:
            reps[r].append(f1w(pred["cs_fof"], w) - f1w(pred[r], w))
    one = np.ones(len(yy))
    T["f1_global_cs_fof"] = f1w(pred["cs_fof"], one)
    T["dif_f1_global"] = [f1w(pred["cs_fof"], one) - f1w(pred[r], one) for r in T.rival]
    T["ic95_bajo"] = [np.quantile(reps[r], 0.025) for r in T.rival]
    T["ic95_alto"] = [np.quantile(reps[r], 0.975) for r in T.rival]
    T.to_csv(AQUI / "umbral_interno_pruebas.csv", index=False)
    print(T.round(4).to_string(index=False))
    (AQUI / "umbral_interno_f1_global.json").write_text(
        json.dumps({m: round(float(f1w(pred[m], one)), 4) for m in MODELOS}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
