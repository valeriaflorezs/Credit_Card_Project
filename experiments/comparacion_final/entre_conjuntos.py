"""Comparación entre conjuntos de datos (Entregable 3, sección 6.3): Friedman con corrección de
Iman-Davenport sobre el F1 medio de cada modelo en Taiwán (mejor combinación) y en los tres
conjuntos adicionales, y Wilcoxon-Holm de CS-FOF frente a cada modelo.
Uso: python experiments/comparacion_final/entre_conjuntos.py"""
import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, f as fdist, wilcoxon, studentized_range
RAIZ = Path(__file__).resolve().parents[2]
os.chdir(Path(__file__).resolve().parent)
mt = pd.read_csv(RAIZ / "results" / "master_table_e3.csv")
f1 = {"taiwan": mt.groupby("model").f1_mean.max()}
for n in ["german", "australian", "polish"]:
    d = pd.read_csv(RAIZ / "results" / "additional_datasets" / f"{n}.csv")
    f1[n] = d.groupby("model").f1.mean()
T = pd.DataFrame(f1).dropna()
R = T.rank(axis=0, ascending=False)               # rango por conjunto (1 = mejor)
N, k = T.shape[1], T.shape[0]
chi2, p = friedmanchisquare(*T.T.values)
ff = (N - 1) * chi2 / (N * (k - 1) - chi2)        # Iman-Davenport
p_id = fdist.sf(ff, k - 1, (k - 1) * (N - 1))
cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
res = T.assign(rango_medio=R.mean(axis=1)).sort_values("rango_medio")
print(res.round(4).to_string())
print(f"\nFriedman: chi2 = {chi2:.2f}, p = {p:.4f} | Iman-Davenport: F = {ff:.2f}, p = {p_id:.4f} | Nemenyi CD = {cd:.2f}")
filas = []
for mo in T.index:
    if mo == "cs_fof": continue
    try: pw = wilcoxon(T.loc["cs_fof"], T.loc[mo]).pvalue
    except ValueError: pw = 1.0
    filas.append(dict(modelo=mo, gana_cs_fof=int((T.loc["cs_fof"] > T.loc[mo]).sum()), p_wilcoxon=pw))
W = pd.DataFrame(filas)
o = W.p_wilcoxon.sort_values().index
W.loc[o, "p_holm"] = np.minimum(1, np.maximum.accumulate([W.loc[i, "p_wilcoxon"] * (len(o) - j) for j, i in enumerate(o)]))
print("\nCS-FOF frente a cada modelo en los 4 conjuntos (Wilcoxon con Holm):"); print(W.round(4).to_string(index=False))
res.to_csv("entre_conjuntos_f1.csv"); W.to_csv("entre_conjuntos_wilcoxon.csv", index=False)
pd.Series(dict(friedman_chi2=chi2, friedman_p=p, iman_davenport_F=ff, iman_davenport_p=p_id, nemenyi_cd=cd)).to_json("entre_conjuntos_friedman.json", indent=1)
