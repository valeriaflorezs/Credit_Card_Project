"""Pruebas de Friedman, Nemenyi, t de Nadeau-Bengio, Wilcoxon (Holm) y delta de Cliff
sobre la mejor combinación de cada modelo. Uso: python experiments/comparacion_final/pruebas.py"""
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, t as tdist, studentized_range
RAIZ = Path(__file__).resolve().parents[2]
R = str(RAIZ / "results") + "/"
os.chdir(Path(__file__).resolve().parent)  # las salidas quedan junto al script
m = pd.read_csv(R + "master_table_e3.csv")  # 7 modelos base + CS-FOF + LightGBM + CatBoost
m = m.dropna(subset=["f1_per_fold"])
m["f1_mean"] = m.f1_per_fold.apply(lambda s: np.mean(json.loads(s)))
best = m.loc[m.groupby("model").f1_mean.idxmax()].sort_values("f1_mean", ascending=False)
F = {r.model: np.array(json.loads(r.f1_per_fold)) for r in best.itertuples()}
A = {r.model: np.array(json.loads(r.roc_auc_per_fold)) for r in best.itertuples()}
print("Mejor combinación por modelo (F1 de la clase default, 5 pliegues externos):")
for r in best.itertuples():
    print(f"  {r.model:20s} {r.technique:13s} {r.method:16s} F1 {F[r.model].mean():.4f} ± {F[r.model].std(ddof=1):.4f}  AUC {A[r.model].mean():.4f}")
# Friedman + Nemenyi entre los 8 modelos (bloques = pliegues)
mods = list(best.model); M = np.column_stack([F[k] for k in mods])
stat, p = friedmanchisquare(*M.T)
ranks = pd.DataFrame(M, columns=mods).rank(axis=1, ascending=False).mean()
k, n = len(mods), M.shape[0]
q = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2); cd = q * np.sqrt(k * (k + 1) / (6 * n))
print(f"\nFriedman: chi2 = {stat:.2f}, p = {p:.2e} | Nemenyi: diferencia crítica = {cd:.2f}")
print("Rango medio (1 = mejor):", ranks.sort_values().round(2).to_dict())
# CS-FOF frente a cada modelo
cs = F["cs_fof"]; filas = []
for mo in mods:
    if mo == "cs_fof": continue
    d = cs - F[mo]
    tt = d.mean() / np.sqrt((1 / n + 1 / 4) * d.var(ddof=1)); p_nb = 2 * tdist.sf(abs(tt), n - 1)
    p_w = wilcoxon(cs, F[mo]).pvalue
    cliff = np.mean([np.sign(a - b) for a in cs for b in F[mo]])
    filas.append(dict(modelo=mo, dif_f1=d.mean(), gana=int((d > 0).sum()), p_nadeau_bengio=p_nb, p_wilcoxon=p_w, cliff=cliff))
T = pd.DataFrame(filas)
for c in ["p_nadeau_bengio", "p_wilcoxon"]:  # Holm
    o = T[c].sort_values().index; adj = np.minimum(1, np.maximum.accumulate([T.loc[i, c] * (len(o) - j) for j, i in enumerate(o)]))
    T.loc[o, c + "_holm"] = adj
print("\nCS-FOF frente a la mejor combinación de cada modelo:")
print(T.round(4).to_string(index=False))
best.to_csv("mejor_por_modelo.csv", index=False); T.to_csv("cs_fof_vs_modelos.csv", index=False)
json.dump(dict(friedman_chi2=stat, friedman_p=p, nemenyi_cd=cd, rangos=ranks.to_dict()), open("friedman.json", "w"), indent=1)
