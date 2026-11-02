"""DENTRO de la VM: compara KNN en GPU (cuML) con KNN de scikit-learn en un fold real."""
import os, sys, json
os.chdir("/content/credit_card"); sys.path.insert(0, "/content/credit_card")
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.data_loader import load_processed_dataset, split_features_target
from src.preprocessing import remove_duplicate_rows, get_preprocessing_steps
from src.models import MODEL_REGISTRY, BalancedKNeighborsClassifier, CuMLBalancedKNN
X, y = split_features_target(remove_duplicate_rows(load_processed_dataset())[0])
tr, te = next(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
steps = get_preprocessing_steps(MODEL_REGISTRY["knn"])
from sklearn.pipeline import Pipeline
pre = Pipeline(steps).fit(X.iloc[tr], y.iloc[tr])
Xtr, Xte, ytr = pre.transform(X.iloc[tr]), pre.transform(X.iloc[te]), y.iloc[tr].to_numpy()
if hasattr(Xtr, "toarray"): Xtr, Xte = Xtr.toarray(), Xte.toarray()
for params in ({"n_neighbors": 15, "weights": "distance", "p": 2}, {"n_neighbors": 35, "weights": "uniform", "p": 1},
               {"n_neighbors": 7, "weights": "distance", "p": 1, "class_weight": "balanced"},
               {"n_neighbors": 51, "weights": "uniform", "p": 2, "class_weight": "balanced"}):
    a = BalancedKNeighborsClassifier(**params).fit(Xtr, ytr)
    b = CuMLBalancedKNN(**params).fit(Xtr, ytr)
    pa, pb = a.predict_proba(Xte), b.predict_proba(Xte)
    print(params, "max|dproba|=%.4f" % np.abs(pa - pb).max(), "acuerdo predict=%.5f" % (a.predict(Xte) == b.predict(Xte)).mean())
