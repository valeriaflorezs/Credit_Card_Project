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
 "RF (mejores hiperparámetros pliegue 1)": lambda: RandomForestClassifier(235, max_depth=11, max_features="log2", min_samples_leaf=1, min_samples_split=45, class_weight="balanced", random_state=42, n_jobs=4),
 "XGBoost (mejores hiperparámetros pliegue 1)": lambda: XGBClassifier(n_estimators=192, max_depth=8, learning_rate=0.0154, subsample=0.598, colsample_bytree=0.837, reg_lambda=0.632, scale_pos_weight=spw, random_state=42, n_jobs=4, tree_method="hist"),
 "V0 CS-FOF base": lambda: CSFisherObliqueForest(**base),
 "V1 + oblicuo con p/2 variables": lambda: CSFisherObliqueForest(**base, oblique_features=0.5),
 "V2 + 2 candidatos oblicuos": lambda: CSFisherObliqueForest(**base, n_oblique=2),
 "V3 + p/2 variables y 2 candidatos": lambda: CSFisherObliqueForest(**base, oblique_features=0.5, n_oblique=2),
 "V4 sin corte oblicuo (ablación)": lambda: CSFisherObliqueForest(**base, oblique=False),
 "V5 = V1 más profundo (d=14, h=10)": lambda: CSFisherObliqueForest(**{**base, "max_depth": 14, "min_samples_leaf": 10}, oblique_features=0.5),
 "V6 = V1 con 128 bins": lambda: CSFisherObliqueForest(**base, oblique_features=0.5, n_bins=128),
}
res = {}
for nombre, mk in var.items():
    f1s, aucs, t0 = [], [], time.time()
    for a, b in interna:
        m = mk().fit(Xd[a], yd[a]); p = m.predict_proba(Xd[b])[:, 1]
        f1s.append(f1_score(yd[b], p > 0.5)); aucs.append(roc_auc_score(yd[b], p))
    res[nombre] = dict(f1=f1s, auc=aucs, seg=time.time() - t0)
    print(f"{nombre:45s} F1 {np.mean(f1s):.4f}±{np.std(f1s):.4f}  AUC {np.mean(aucs):.4f}  {time.time()-t0:5.0f}s", flush=True)
json.dump(res, open("ronda1_resultados.json", "w"), indent=1)
