import time, numpy as np
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
X, y = make_classification(n_samples=24000, n_features=25, n_informative=10, weights=[0.78], random_state=0)
X = X.astype("float32")
for dev in ("cuda", "cpu"):
    for params in ({"n_estimators": 100, "max_depth": 4}, {"n_estimators": 300, "max_depth": 8}):
        ts = []
        for _ in range(3):
            m = XGBClassifier(device=dev, tree_method="hist", n_jobs=2, eval_metric="logloss", **params)
            t = time.perf_counter(); m.fit(X, y); m.predict_proba(X[:6000]); ts.append(time.perf_counter() - t)
        print(dev, params, "mediana %.2fs" % sorted(ts)[1])
