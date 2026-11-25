import glob, os, sys, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from combos import thr_fast, f1_at
from hib import hib_scores
PRE = sys.argv[1] if len(sys.argv) > 1 else "tw2"; SEEDS = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "7100,7101,7102,7103,7104").split(",")]
BENCH = ["xgb", "rf", "svm", "knn", "lr", "dt", "nb"]
def delong(y, a, b):
    m = int((y == 1).sum()); n = len(y) - m
    def comp(s):
        p, q = s[y == 1], s[y == 0]; r = stats.rankdata(np.concatenate([p, q])); rp = stats.rankdata(p); rq = stats.rankdata(q)
        return (r[:m].sum()-m*(m+1)/2)/(m*n), (r[:m]-rp)/n, 1-(r[m:]-rq)/m
    A1, v10a, v01a = comp(a); A2, v10b, v01b = comp(b); S = np.cov(np.vstack([v10a, v10b]))/m + np.cov(np.vstack([v01a, v01b]))/n
    var = S[0,0]+S[1,1]-2*S[0,1]; z = (A1-A2)/np.sqrt(var); return A1-A2, 2*stats.norm.sf(abs(z))
def holm(p):
    o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for i, idx in enumerate(o): run = max(run, (len(p)-i)*p[idx]); adj[idx] = min(1, run)
    return adj
def cliff(x, y): return (np.sum(x[:, None] > y[None, :]) - np.sum(x[:, None] < y[None, :]))/(len(x)*len(y))
F1 = {}; Y = {}; SC = {}; BP = {}; J = 0
for seed in SEEDS:
    for f in range(5):
        k = f"s{seed}_f{f}.npz"; fm, fe, fs = f"{PRE}_members/{k}", f"{PRE}_extra/{k}", f"{PRE}_svm/{k}"
        if not (os.path.exists(fm) and os.path.exists(fe) and os.path.exists(fs)): print("FALTA", k); sys.exit(1)
        m, e, s = [dict(np.load(x, allow_pickle=True)) for x in (fm, fe, fs)]; names = list(m["models"]); y, yt = m["ytr"], m["yte"]; J += 1
        D = {}; 
        for n in ["xgb", "rf", "lr"]: j = names.index(n); D[n] = (m["O"][:, j], m["T"][:, j])
        for j, n in enumerate(list(e["models"])): D[n] = (e["O"][:, j], e["T"][:, j])
        D["svm"] = (s["O"][:, 0], s["T"][:, 0]); so, st, _ = hib_scores(m["O"], y, m["T"]); D["HIB"] = (so, st)
        for n, (o, t) in D.items():
            th = thr_fast(y, o); F1.setdefault(n, []).append(f1_at(yt, t, th)); SC.setdefault((n, seed), []).append(t); BP.setdefault((n, seed), []).append(t >= th)
        Y.setdefault(seed, []).append(yt)
F1 = pd.DataFrame(F1); print(f"J = {J} pliegues"); print(F1.mean().round(4).sort_values(ascending=False).to_string())
rows = []
for b in BENCH:
    x = (F1["HIB"]-F1[b]).values; pw = stats.wilcoxon(F1["HIB"], F1[b]).pvalue; pd_, pm = [], []
    for seed in SEEDS:
        y = np.concatenate(Y[seed]); a_, b_ = np.concatenate(SC[("HIB", seed)]), np.concatenate(SC[(b, seed)]); pd_.append(delong(y, a_, b_)[1])
        ca, cb = np.concatenate(BP[("HIB", seed)]) == y, np.concatenate(BP[(b, seed)]) == y; n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum()); pm.append(stats.binomtest(n01, n01+n10, .5).pvalue if n01+n10 else 1.0)
    se = np.sqrt(1/J+.25)*x.std(ddof=1)
    rows.append(dict(bench=b, dif_F1=x.mean(), gana=f"{int((x>0).sum())}/{J}", cliff=cliff(F1["HIB"].values, F1[b].values), p_P1_wilcoxon=pw, p_P2_delong_mediana=float(np.median(pd_)), p_P3_mcnemar_mediana=float(np.median(pm)),
                     p_NB_informativo=2*stats.t.sf(abs(x.mean()/se), J-1)))
T = pd.DataFrame(rows)
for c in ["p_P1_wilcoxon", "p_P2_delong_mediana", "p_P3_mcnemar_mediana"]: T[c.replace("p_", "holm_")] = holm(T[c].values)
cols = BENCH + ["HIB"]; Rk = F1[cols].rank(axis=1, ascending=False).mean(); k = len(cols); fr = stats.friedmanchisquare(*[F1[c] for c in cols]); se = np.sqrt(k*(k+1)/(6*J))
T["p_nemenyi_informativo"] = [float(stats.studentized_range.sf(abs(Rk["HIB"]-Rk[b])/se*np.sqrt(2), k, 1e6)) for b in BENCH]
T["superior"] = np.where((T.dif_F1 > 0) & (T.holm_P1_wilcoxon < .05) & (T.holm_P2_delong_mediana < .05) & (T.holm_P3_mcnemar_mediana < .05) & (T.cliff.abs() >= .147) & (fr.pvalue < .05), "SI", "NO")
print(f"\nP4 Friedman global: chi2={fr.statistic:.2f}, p={fr.pvalue:.2e}; rangos: {Rk.round(2).to_dict()}"); print(T.round(4).to_string(index=False))
print("\nR3 Taiwán (superior a los 7):", "CUMPLE" if (T.superior == "SI").all() else "NO CUMPLE"); T.to_csv(f"conf2_{PRE}_contrastes.csv", index=False); F1.to_csv(f"conf2_{PRE}_f1.csv", index=False)
