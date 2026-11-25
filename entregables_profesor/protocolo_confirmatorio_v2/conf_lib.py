import warnings; warnings.filterwarnings("ignore")
import os, sys; sys.path.insert(0, "/home/user/Credit_Card_Project")
from common2 import *
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from src import preprocessing as P
MEMBERS = ["cs_fof", "xgb", "cat", "lgbm", "rf", "et", "hgb", "lr"]; BENCH_EXTRA = ["dt", "knn", "nb"]
def build_features():
    cache = "features_conf.pkl"
    if os.path.exists(cache): return pd.read_pickle(cache)
    raw = P.load_raw_dataset(); raw, _ = P.regroup_undocumented_categories(raw); raw, _ = P.remove_duplicate_rows(raw)
    X, y = load(); assert len(raw) == len(X) and (raw["PAY_0"].values == X["PAY_0"].values).all() and (raw["default_payment_next_month"].values == y).all()
    L = raw["LIMIT_BAL"].values.astype(float); B = np.c_[[raw[f"BILL_AMT{i}"].values for i in range(1, 7)]].T.astype(float); A = np.c_[[raw[f"PAY_AMT{i}"].values for i in range(1, 7)]].T.astype(float)
    F = pd.DataFrame(index=X.index); U = np.clip(B/L[:, None], -1, 3)
    for i in range(6): F[f"UTIL_{i+1}"] = U[:, i]
    F["UTIL_MEAN"] = U.mean(1); F["UTIL_MAX"] = U.max(1); F["N_SOBRE_LIMITE"] = (B > L[:, None]).sum(1)
    R = np.full((len(L), 5), np.nan)
    for i in range(5):
        ok = B[:, i+1] > 0; R[ok, i] = np.clip(A[ok, i]/B[ok, i+1], 0, 2)
    for i in range(5): F[f"RATIO_PAGO_{i+1}"] = R[:, i]
    F["RATIO_PAGO_MEAN"] = np.nanmean(R, 1); F["RATIO_PAGO_MIN"] = np.nanmin(R, 1)
    F["N_PAGO_MINIMO"] = np.nansum(R < 0.10, 1); F["N_PAGO_TOTAL"] = np.nansum(R >= 0.98, 1); F["N_SIN_PAGO"] = (A[:, :5] == 0).sum(1)
    F["PAGOS_SOBRE_FACTURAS"] = np.clip(A.sum(1)/np.maximum(np.clip(B, 0, None).sum(1), 1), 0, 3)
    F["CAMBIO_UTIL"] = np.clip((B[:, 0]-B[:, 5])/L, -3, 3); F["CAMBIO_UTIL_RECIENTE"] = np.clip((B[:, 0]-B[:, 1])/L, -3, 3)
    Pm = X[PAYS].values; run = np.zeros(len(X))
    for i in range(len(X)):
        k = 0
        for v in Pm[i]:
            if v > 0: k += 1
            else: break
        run[i] = k
    F["RACHA_ATRASO"] = run; F["N_MESES_ATRASO"] = (Pm > 0).sum(1); F["MAX_ATRASO"] = Pm.max(1)
    XF = pd.concat([X, F], axis=1).fillna(-1.0); pd.to_pickle((XF, y), cache); return XF, y
def build_bench_extra(n):
    if n == "dt": return DecisionTreeClassifier(max_depth=6, min_samples_split=20, min_samples_leaf=14, class_weight="balanced", random_state=0)
    if n == "knn": return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=41, p=2, weights="uniform", n_jobs=2))
    if n == "nb": return make_pipeline(StandardScaler(), GaussianNB(var_smoothing=1e-6))
    if n == "svm": return make_pipeline(StandardScaler(), SVC(C=5.456725485601478, gamma=0.005762487216478602, class_weight="balanced", cache_size=1500, random_state=0))
def score_of(m, X):
    return m.decision_function(X) if hasattr(m, "decision_function") and not hasattr(m, "predict_proba") else m.predict_proba(X)[:, 1]
