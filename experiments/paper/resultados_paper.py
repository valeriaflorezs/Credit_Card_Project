"""Análisis que pide la guía de redacción de resultados del paper (comparación con todos los baselines del curso).

Usa SIEMPRE los mismos 5 pliegues externos (StratifiedKFold, semilla 42) que la validación anidada.

Etapas (python experiments/paper/resultados_paper.py [etapa ...]):
- ``lr``: tres variantes de regresión logística que la guía pide por separado, con validación anidada 5 × 5
  y pesos de clase: sin penalización, Ridge (L2) y Lasso (L1). En Ridge y Lasso el bucle interno busca C con
  25 configuraciones aleatorias (log-uniforme en [1e-4, 1e2]) y F1, como el resto de los modelos.
- ``oof``: probabilidades fuera de pliegue de SVM, KNN, árbol de decisión, regresión logística y Naive Bayes
  (mejor combinación de cada uno, hiperparámetros de cada pliegue) para la prueba de McNemar exacta.
- ``ablacion``: variantes de CS-FOF en los 5 pliegues externos con los hiperparámetros de cada pliegue:
  (a) sin corte oblicuo, (b) sin costo (sin pesos de clase), (d) dirección con todas las variables del nodo.
- ``tabla``: métricas por pliegue (F1, AUC-ROC, AUC-PR, MCC, exactitud balanceada) de todos los modelos.
- ``pruebas``: t corregida de Nadeau–Bengio, McNemar exacta y t correlacionada bayesiana con ROPE para
  CS-FOF frente a cada baseline, con Holm; tamaño del efecto A12 de Vargha–Delaney.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
                             roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ / "experiments" / "robustez"))
from ampliada import _datos, _params  # noqa: E402
from robustez import fabrica  # noqa: E402

CF = RAIZ / "experiments" / "comparacion_final"
OOF5 = CF / "probabilidades_oof.npz"
OOF_EXTRA = AQUI / "probabilidades_oof_extra.npz"
SEED = 42
ROPE = 0.01


def metricas(y, p, u=0.5):
    pr = p >= u
    return dict(f1=f1_score(y, pr), roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                mcc=matthews_corrcoef(y, pr), bal_acc=balanced_accuracy_score(y, pr))


# --------------------------------------------------------------------- regresión logística (3 variantes)
def lr():
    X, y, _, folds = _datos()
    filas, oof = [], {v: np.zeros(len(y)) for v in ("lr_none", "ridge", "lasso")}
    for var, pen in (("lr_none", None), ("ridge", "l2"), ("lasso", "l1")):
        for k, (tr, te) in enumerate(folds):
            t = time.time()
            base = Pipeline([("sc", StandardScaler()),
                             ("lr", LogisticRegression(penalty=pen, solver="saga", max_iter=5000, class_weight="balanced",
                                                       random_state=SEED))])
            if pen is None:
                mod = base.fit(X[tr], y[tr]); C = None
            else:
                bus = RandomizedSearchCV(base, {"lr__C": loguniform(1e-4, 1e2)}, n_iter=25, scoring="f1",
                                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED), random_state=SEED, n_jobs=4)
                mod = bus.fit(X[tr], y[tr]).best_estimator_; C = float(bus.best_params_["lr__C"])
            p = mod.predict_proba(X[te])[:, 1]; oof[var][te] = p
            filas.append(dict(model=var, fold=k, C=C, seconds=round(time.time() - t, 1), **metricas(y[te], p)))
            print(var, k, round(filas[-1]["f1"], 4), C, flush=True)
    pd.DataFrame(filas).to_csv(AQUI / "lr_variantes_pliegues.csv", index=False)
    _guardar_oof(oof)


def _guardar_oof(nuevos):
    prev = dict(np.load(OOF_EXTRA)) if OOF_EXTRA.exists() else {}
    prev.update(nuevos); np.savez(OOF_EXTRA, **prev)


# --------------------------------------------------------------------- OOF de los 5 baselines restantes
def oof():
    from src.balancing import build_pipeline
    from src.data_loader import split_features_target
    from src.models import MODEL_REGISTRY
    from src.preprocessing import build_final_dataset, get_preprocessing_steps, load_raw_dataset, remove_duplicate_rows
    _, y, _, folds = _datos()
    df, _ = build_final_dataset(load_raw_dataset()); df, _ = remove_duplicate_rows(df)
    Xp, yp = split_features_target(df)
    assert (yp.to_numpy() == y).all()
    best = pd.read_csv(CF / "mejor_por_modelo.csv").set_index("model")
    nuevos = {}
    for m in ("naive_bayes", "logistic_regression", "decision_tree", "knn", "svm"):
        tec = best.loc[m, "technique"]; P = json.loads(best.loc[m, "best_params_per_fold"]); spec = MODEL_REGISTRY[m]
        p = np.zeros(len(y)); t = time.time()
        for k, (tr, te) in enumerate(folds):
            pipe = build_pipeline(spec, tec, y_train=yp.iloc[tr], preprocessing_steps=get_preprocessing_steps(spec),
                                  random_state=SEED, **P[k])
            pipe.fit(Xp.iloc[tr], yp.iloc[tr]); p[te] = pipe.predict_proba(Xp.iloc[te])[:, 1]
        nuevos[m] = p
        print(m, tec, "F1 OOF", round(f1_score(y, p >= 0.5), 4), f"{time.time() - t:.0f}s", flush=True)
        _guardar_oof({m: p})


# --------------------------------------------------------------------- ablación
def ablacion():
    X, y, _, folds = _datos(); P = _params()
    full = np.load(OOF5)["cs_fof"]
    variantes = {"sin_oblicuo": dict(oblique=False), "sin_costo": dict(class_weight=None),
                 "todas_las_variables": dict(oblique_features=1.0)}
    filas = [dict(variante="completo", fold=k, **metricas(y[te], full[te])) for k, (_, te) in enumerate(folds)]
    oofv = {}
    for v, cambio in variantes.items():
        p = np.zeros(len(y))
        for k, (tr, te) in enumerate(folds):
            t = time.time()
            mod = fabrica("cs_fof", P["cs_fof"][k], y[tr]); mod.set_params(**cambio)
            p[te] = mod.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            filas.append(dict(variante=v, fold=k, seconds=round(time.time() - t, 1), **metricas(y[te], p[te])))
            print(v, k, round(filas[-1]["f1"], 4), round(filas[-1]["roc_auc"], 4), flush=True)
        oofv[f"abl_{v}"] = p
    pd.DataFrame(filas).to_csv(AQUI / "ablacion_pliegues.csv", index=False)
    _guardar_oof(oofv)


# --------------------------------------------------------------------- tabla principal
NOMBRE = {"cs_fof": "CS-FOF (proposed)", "catboost": "CatBoost", "xgboost": "XGBoost", "lightgbm": "LightGBM",
          "random_forest": "Random Forest", "svm": "SVM (RBF)", "knn": "KNN", "decision_tree": "Decision tree",
          "logistic_regression": "Logistic regression (tuned L1/L2)", "naive_bayes": "Naive Bayes",
          "lr_none": "Logistic regression (no penalty)", "ridge": "Ridge (L2 logistic)", "lasso": "Lasso (L1 logistic)"}


def _conteos_por_pliegue(y, folds):
    return [(int(y[te].sum()), int((y[te] == 0).sum())) for _, te in folds]


def tabla():
    _, y, _, folds = _datos()
    best = pd.read_csv(CF / "mejor_por_modelo.csv").set_index("model")
    post = pd.read_csv(RAIZ / "results" / "experiments_posthoc_metrics.csv")
    oof5 = np.load(OOF5)
    filas = []
    for m in best.index:
        r = best.loc[m]
        f1 = json.loads(r.f1_per_fold); pr = json.loads(r.precision_per_fold); rc = json.loads(r.recall_per_fold)
        auc = json.loads(r.roc_auc_per_fold)
        if m in oof5.files:
            ap = [average_precision_score(y[te], oof5[m][te]) for _, te in folds]
        elif isinstance(r.pr_auc_per_fold, str):
            ap = json.loads(r.pr_auc_per_fold)
        else:
            ap = [np.nan] * 5
        for k, (P_, N_) in enumerate(_conteos_por_pliegue(y, folds)):
            tp = rc[k] * P_; fp = tp / pr[k] - tp; fn = P_ - tp; tn = N_ - fp
            mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            filas.append(dict(model=m, fold=k, f1=f1[k], roc_auc=auc[k], pr_auc=ap[k], mcc=mcc,
                              bal_acc=0.5 * (rc[k] + tn / N_)))
    lrv = pd.read_csv(AQUI / "lr_variantes_pliegues.csv")
    for _, r in lrv.iterrows():
        filas.append(dict(model=r.model, fold=int(r.fold), f1=r.f1, roc_auc=r.roc_auc, pr_auc=r.pr_auc, mcc=r.mcc, bal_acc=r.bal_acc))
    T = pd.DataFrame(filas); T.to_csv(AQUI / "metricas_pliegues.csv", index=False)
    R = T.groupby("model")[["f1", "roc_auc", "pr_auc", "mcc", "bal_acc"]].agg(["mean", "std"])
    R = R.sort_values(("f1", "mean"), ascending=False); R.to_csv(AQUI / "tabla_principal.csv")
    print(R.round(4).to_string())


# --------------------------------------------------------------------- pruebas
def nadeau_bengio(a, b, k=5):
    d = np.asarray(a) - np.asarray(b); n = len(d)
    t = d.mean() / np.sqrt((1 / n + 1 / (k - 1)) * d.var(ddof=1))
    return float(t), float(2 * stats.t.sf(abs(t), n - 1))


def bayes_correlado(a, b, rope=ROPE, k=5):
    """t correlacionada bayesiana (Corani y Benavoli, 2015): posterior t de Student con correlación rho = 1/k."""
    d = np.asarray(a) - np.asarray(b); n = len(d); rho = 1 / k
    esc = np.sqrt((1 / n + rho / (1 - rho)) * d.var(ddof=1))
    post = stats.t(df=n - 1, loc=d.mean(), scale=esc)
    return float(post.cdf(-rope)), float(post.cdf(rope) - post.cdf(-rope)), float(1 - post.cdf(rope))


def mcnemar_exacta(y, pa, pb):
    a, b = (pa == y), (pb == y)
    n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())   # solo acierta CS-FOF / solo acierta el rival
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue
    return n01, n10, float(p)


def a12(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean([(x > z) + 0.5 * (x == z) for x in a for z in b]))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); prev = 0
    for j, i in enumerate(o):
        prev = max(prev, min(1, (len(p) - j) * p[i])); adj[i] = prev
    return adj


def pruebas():
    _, y, _, folds = _datos()
    T = pd.read_csv(AQUI / "metricas_pliegues.csv")
    F = {m: g.sort_values("fold").f1.to_numpy() for m, g in T.groupby("model")}
    oof = dict(np.load(OOF5)); oof.update(dict(np.load(OOF_EXTRA)))
    yb = y.astype(int); cs = oof["cs_fof"] >= 0.5
    filas = []
    for m in F:
        if m == "cs_fof":
            continue
        t, p_t = nadeau_bengio(F["cs_fof"], F[m])
        pl, pr_, pw = bayes_correlado(F["cs_fof"], F[m])
        n01, n10, p_mc = mcnemar_exacta(yb, cs, oof[m] >= 0.5) if m in oof else (np.nan, np.nan, np.nan)
        d = F["cs_fof"] - F[m]
        lo, hi = stats.t.interval(0.95, 4, loc=d.mean(), scale=np.sqrt((1 / 5 + 1 / 4) * d.var(ddof=1)))
        filas.append(dict(baseline=m, dif_f1=d.mean(), ic95_bajo=lo, ic95_alto=hi, t_nb=t, p_nb=p_t,
                          mcnemar_solo_cs=n01, mcnemar_solo_rival=n10, p_mcnemar=p_mc,
                          p_bayes_cs_peor=pl, p_bayes_rope=pr_, p_bayes_cs_mejor=pw, a12=a12(F["cs_fof"], F[m])))
    R = pd.DataFrame(filas)
    R["p_nb_holm"] = holm(R.p_nb); R["p_mcnemar_holm"] = holm(R.p_mcnemar.fillna(1))
    R = R.sort_values("dif_f1"); R.to_csv(AQUI / "pruebas_cs_fof.csv", index=False)
    print(R.round(4).to_string(index=False))
    # ablación: completo frente a cada variante
    A = pd.read_csv(AQUI / "ablacion_pliegues.csv")
    Fa = {v: g.sort_values("fold") for v, g in A.groupby("variante")}
    ab = []
    for v in Fa:
        if v == "completo":
            continue
        row = dict(variante=v)
        for met in ("f1", "roc_auc", "pr_auc", "mcc"):
            d = Fa["completo"][met].to_numpy() - Fa[v][met].to_numpy()
            row[f"{met}_completo"] = Fa["completo"][met].mean(); row[f"{met}_variante"] = Fa[v][met].mean()
            row[f"dif_{met}"] = d.mean(); row[f"p_nb_{met}"] = nadeau_bengio(Fa["completo"][met], Fa[v][met])[1]
        pl, pr_, pw = bayes_correlado(Fa["completo"].f1, Fa[v].f1)
        row.update(p_bayes_completo_mejor=pw, p_bayes_rope=pr_, p_bayes_completo_peor=pl)
        ab.append(row)
    AB = pd.DataFrame(ab); AB["p_nb_f1_holm"] = holm(AB.p_nb_f1); AB["p_nb_roc_auc_holm"] = holm(AB.p_nb_roc_auc)
    AB.to_csv(AQUI / "ablacion_pruebas.csv", index=False); print(AB.round(4).to_string(index=False))


if __name__ == "__main__":
    for etapa in sys.argv[1:] or ["lr", "ablacion", "oof", "tabla", "pruebas"]:
        globals()[etapa]()
