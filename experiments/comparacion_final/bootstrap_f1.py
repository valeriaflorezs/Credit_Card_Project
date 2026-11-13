"""Intervalos de confianza bootstrap de la diferencia de F1 entre CS-FOF y cada rival (Entregable 3, 6.5 y 6.7).

Datos: probabilidades fuera de pliegue de los 29 965 clientes (``probabilidades_oof.npz``, reajuste con los
hiperparámetros elegidos en cada pliegue externo), umbral 0.5, F1 de la clase de impago.

Bootstrap pareado: se remuestrean CLIENTES con reemplazo, estratificando por clase, y en cada réplica se
calcula F1(CS-FOF) − F1(rival) sobre los mismos clientes. B = 10 000 réplicas, semilla 42.

- Intervalo percentil y BCa (sesgo corregido y acelerado, recomendado por la guía para métricas
  asimétricas). La aceleración sale del jackknife; como F1 depende solo de los conteos de la matriz de
  confusión, el jackknife se calcula exacto y rápido por tipo de cliente (16 combinaciones de aciertos y
  errores de los dos modelos), sin recalcular 29 965 veces.
- p bootstrap bilateral (2 × la fracción menor de réplicas con diferencia ≤ 0 o ≥ 0) y p ajustado con Holm
  sobre los 4 rivales.

Alcance: el intervalo es CONDICIONAL a los modelos ya entrenados; mide la incertidumbre por la muestra de
clientes, no la variación por volver a entrenar (esa la recogen los pliegues y las 10 semillas).
Uso (desde la raíz): python experiments/comparacion_final/bootstrap_f1.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

AQUI = Path(__file__).resolve().parent
RIVALES = ["catboost", "xgboost", "lightgbm", "random_forest"]
B, SEMILLA, ALFA = 10_000, 42, 0.05


def f1_conteos(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn)


def diferencia(ya, yb, y, w):
    """F1(a) − F1(b) con pesos (número de veces que cada cliente aparece en la réplica)."""
    def f1(p):
        tp = (w * (p & (y == 1))).sum(); fp = (w * (p & (y == 0))).sum(); fn = (w * (~p & (y == 1))).sum()
        return f1_conteos(tp, fp, fn)
    return f1(ya) - f1(yb)


def jackknife(ya, yb, y):
    """Valores leave-one-out de la diferencia, agrupados por tipo de cliente (exacto)."""
    def conteos(p):
        return (p & (y == 1)).sum(), (p & (y == 0)).sum(), (~p & (y == 1)).sum()
    ca, cb = conteos(ya), conteos(yb)
    tipo = ya.astype(int) * 4 + yb.astype(int) * 2 + y.astype(int)
    vals, pesos = [], []
    for t in np.unique(tipo):
        a, b, yy = (t >> 2) & 1, (t >> 1) & 1, t & 1
        da = (a and yy, a and not yy, (not a) and yy)   # qué conteo pierde el modelo a al sacar este cliente
        db = (b and yy, b and not yy, (not b) and yy)
        fa = f1_conteos(*(c - int(d) for c, d in zip(ca, da)))
        fb = f1_conteos(*(c - int(d) for c, d in zip(cb, db)))
        vals.append(fa - fb); pesos.append((tipo == t).sum())
    return np.array(vals), np.array(pesos)


def bca(theta, reps, jk_vals, jk_pesos):
    z0 = norm.ppf((reps < theta).mean() + 0.5 * (reps == theta).mean())
    m = np.average(jk_vals, weights=jk_pesos)
    d = m - jk_vals
    a = (jk_pesos * d ** 3).sum() / (6 * ((jk_pesos * d ** 2).sum()) ** 1.5)
    lims = []
    for q in (ALFA / 2, 1 - ALFA / 2):
        z = norm.ppf(q)
        lims.append(np.quantile(reps, norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))))
    return lims, float(a), float(z0)


def main():
    d = np.load(AQUI / "probabilidades_oof.npz")
    y = d["y"].astype(int)
    pred = {m: d[m] >= 0.5 for m in ["cs_fof"] + RIVALES}
    rng = np.random.default_rng(SEMILLA)
    idx = {c: np.flatnonzero(y == c) for c in (0, 1)}
    reps = {r: np.empty(B) for r in RIVALES}
    for b in range(B):   # réplica estratificada: mismo número de clientes de cada clase
        w = sum(np.bincount(rng.choice(idx[c], len(idx[c]), replace=True), minlength=len(y)) for c in (0, 1))
        for r in RIVALES:
            reps[r][b] = diferencia(pred["cs_fof"], pred[r], y, w)
    filas = []
    for r in RIVALES:
        theta = diferencia(pred["cs_fof"], pred[r], y, np.ones(len(y)))
        reps_r = reps[r]
        jv, jw = jackknife(pred["cs_fof"], pred[r], y)
        (lo_b, hi_b), a, z0 = bca(theta, reps_r, jv, jw)
        p = min(1.0, 2 * min((reps_r <= 0).mean(), (reps_r >= 0).mean()))
        filas.append(dict(rival=r, f1_cs_fof=diferencia(pred["cs_fof"], np.zeros(len(y), bool), y, np.ones(len(y))),
                          dif_f1=theta, ic95_percentil_bajo=np.quantile(reps_r, ALFA / 2),
                          ic95_percentil_alto=np.quantile(reps_r, 1 - ALFA / 2), ic95_bca_bajo=lo_b, ic95_bca_alto=hi_b,
                          aceleracion=a, sesgo_z0=z0, cs_fof_gana_en=float((reps_r > 0).mean()), p_bootstrap=p))
    T = pd.DataFrame(filas)
    orden = np.argsort(T.p_bootstrap.values)
    holm = np.empty(len(T)); prev = 0
    for k, i in enumerate(orden):
        prev = max(prev, min(1, (len(T) - k) * T.p_bootstrap.values[i])); holm[i] = prev
    T["p_holm"] = holm
    T.to_csv(AQUI / "bootstrap_f1.csv", index=False)
    print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
