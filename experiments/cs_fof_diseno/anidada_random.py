"""Validación anidada 5x5 de CS-FOF con búsqueda aleatoria (25 configuraciones) y class_weight,
igual que el protocolo del proyecto (mismos pliegues externos, semilla 42, métrica F1)."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raíz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import randint, loguniform
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
from datos import cargar
from src.original_model import CSFisherObliqueForest
X, y, cols, folds = cargar()
dist = {"n_estimators": randint(100, 501), "max_depth": randint(6, 15), "min_samples_leaf": randint(10, 61),
        "max_features": ["sqrt", "log2"], "eps": loguniform(0.01, 1.0)}
out = []
for k, (tr, te) in enumerate(folds):
    t0 = time.time()
    inner = StratifiedKFold(5, shuffle=True, random_state=42)
    s = RandomizedSearchCV(CSFisherObliqueForest(class_weight="balanced", random_state=42, n_jobs=4), dist,
                           n_iter=25, scoring="f1", cv=inner, random_state=42, n_jobs=1, refit=True)
    s.fit(X[tr], y[tr])
    p = s.predict_proba(X[te])[:, 1]; yh = (p > 0.5).astype(int)
    r = dict(fold=k, f1=f1_score(y[te], yh), precision=precision_score(y[te], yh), recall=recall_score(y[te], yh),
             roc_auc=roc_auc_score(y[te], p), pr_auc=average_precision_score(y[te], p), inner_best=s.best_score_,
             params={a: (float(b) if isinstance(b, (float, np.floating)) else (int(b) if isinstance(b, (int, np.integer)) else b)) for a, b in s.best_params_.items()},
             oblicuos=s.best_estimator_.fraccion_cortes_oblicuos(), seg=time.time() - t0)
    out.append(r); json.dump(out, open("anidada_random.json", "w"), indent=1)
    print(f"pliegue {k+1}: F1 {r['f1']:.4f} AUC {r['roc_auc']:.4f} (interno {r['inner_best']:.4f}) {r['seg']/60:.1f} min {r['params']}", flush=True)
f = [r["f1"] for r in out]; a = [r["roc_auc"] for r in out]
print(f"FINAL F1 {np.mean(f):.4f} ± {np.std(f, ddof=1):.4f} | AUC {np.mean(a):.4f} ± {np.std(a, ddof=1):.4f}")
