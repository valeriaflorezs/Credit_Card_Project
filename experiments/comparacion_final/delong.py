"""Probabilidades fuera de pliegue (reajuste con los hiperparámetros elegidos en cada pliegue externo),
y pruebas de DeLong (AUC) y McNemar (errores) entre CS-FOF y los dos mejores modelos base."""
import os, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ / "experiments" / "cs_fof_diseno"))
os.chdir(Path(__file__).resolve().parent)
import datos  # usa data/raw/default_of_credit_card_clients.xls; Uso: python experiments/comparacion_final/delong.py
from scipy.stats import norm, chi2
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.original_model import CSFisherObliqueForest
X, y, cols, folds = datos.cargar()
best = pd.read_csv("mejor_por_modelo.csv").set_index("model")
def oof(make):
    p = np.zeros(len(y))
    for k, (tr, te) in enumerate(folds):
        m = make(k, y[tr]).fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p
P = lambda mo: json.loads(best.loc[mo, "best_params_per_fold"])
t = time.time()
prob = {
 "cs_fof": oof(lambda k, yt: CSFisherObliqueForest(class_weight="balanced", random_state=42, n_jobs=4, **P("cs_fof")[k])),
 "xgboost": oof(lambda k, yt: XGBClassifier(random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=4,
                                          scale_pos_weight=(yt == 0).sum() / (yt == 1).sum(), **P("xgboost")[k])),
 "random_forest": oof(lambda k, yt: RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=4, **P("random_forest")[k])),
}
print(f"reajustes en {time.time()-t:.0f}s")
def delong(y, a, b):
    pos, neg = a[y == 1], a[y == 0]; pos2, neg2 = b[y == 1], b[y == 0]
    def comp(p, n):
        v10 = np.array([(x > n).mean() + 0.5 * (x == n).mean() for x in p]); v01 = np.array([(p < x).mean() + 0.5 * (p == x).mean() for x in n])
        return v10.mean(), v10, v01
    A1, v10a, v01a = comp(pos, neg); A2, v10b, v01b = comp(pos2, neg2)
    s10 = np.cov(np.vstack([v10a, v10b])); s01 = np.cov(np.vstack([v01a, v01b]))
    S = s10 / len(pos) + s01 / len(neg); var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (A1 - A2) / np.sqrt(var); return A1, A2, z, 2 * norm.sf(abs(z))
filas = []
from sklearn.metrics import f1_score
for rival in ["xgboost", "random_forest"]:
    A1, A2, z, p = delong(y, prob["cs_fof"], prob[rival])
    ea, eb = (prob["cs_fof"] > 0.5) != y, (prob[rival] > 0.5) != y
    b_, c_ = int((ea & ~eb).sum()), int((~ea & eb).sum())
    stat = (abs(b_ - c_) - 1) ** 2 / (b_ + c_); p_mc = chi2.sf(stat, 1)
    filas.append(dict(rival=rival, auc_cs_fof=A1, auc_rival=A2, delong_z=z, delong_p=p,
                      mcnemar_solo_cs_fof_falla=b_, mcnemar_solo_rival_falla=c_, mcnemar_p=p_mc,
                      f1_oof_cs_fof=f1_score(y, prob["cs_fof"] > 0.5), f1_oof_rival=f1_score(y, prob[rival] > 0.5)))
T = pd.DataFrame(filas); print(T.round(4).to_string(index=False)); T.to_csv("delong_mcnemar.csv", index=False)
np.savez_compressed("probabilidades_oof.npz", y=y, **prob)
