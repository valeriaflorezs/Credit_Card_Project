"""Pruebas de diseño SOLO con el entrenamiento del pliegue externo 1 (validación interna de 3 pliegues).
Los pliegues externos de prueba no se tocan: se reservan para la evaluación final 5x5."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raíz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from datos import cargar
from src.original_model import CSFisherObliqueForest
X, y, cols, folds = cargar()
tr = folds[0][0]; Xd, yd = X[tr], y[tr]
interna = list(StratifiedKFold(3, shuffle=True, random_state=42).split(Xd, yd))
spw = (yd == 0).sum() / (yd == 1).sum()
base = dict(n_estimators=150, max_depth=10, min_samples_leaf=20, class_weight="balanced", random_state=42, n_jobs=4)
var = {
 "R1 base (d=10, h=20, eps=0.1, sqrt, T=150)": lambda: CSFisherObliqueForest(**base),
 "R2 d=8": lambda: CSFisherObliqueForest(**{**base, "max_depth": 8}),
 "R3 h=40": lambda: CSFisherObliqueForest(**{**base, "min_samples_leaf": 40}),
 "R4 eps=0.5": lambda: CSFisherObliqueForest(**{**base, "eps": 0.5}),
 "R5 eps=0.01": lambda: CSFisherObliqueForest(**{**base, "eps": 0.01}),
 "R6 max_features=log2": lambda: CSFisherObliqueForest(**{**base, "max_features": "log2"}),
 "R7 T=300": lambda: CSFisherObliqueForest(**{**base, "n_estimators": 300}),
 "R8 d=12, h=40": lambda: CSFisherObliqueForest(**{**base, "max_depth": 12, "min_samples_leaf": 40}),
}
res = {}
for nombre, mk in var.items():
    f1s, aucs, t0 = [], [], time.time()
    for a, b in interna:
        m = mk().fit(Xd[a], yd[a]); p = m.predict_proba(Xd[b])[:, 1]
        f1s.append(f1_score(yd[b], p > 0.5)); aucs.append(roc_auc_score(yd[b], p))
    res[nombre] = dict(f1=f1s, auc=aucs, seg=time.time() - t0)
    print(f"{nombre:45s} F1 {np.mean(f1s):.4f}±{np.std(f1s):.4f}  AUC {np.mean(aucs):.4f}  {time.time()-t0:5.0f}s", flush=True)
json.dump(res, open("ronda2_resultados.json", "w"), indent=1)
