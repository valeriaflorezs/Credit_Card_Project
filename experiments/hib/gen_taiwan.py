import sys; from conf_lib import *
from sklearn.model_selection import StratifiedKFold
kind = sys.argv[1]; outdir = sys.argv[2]; seeds = [int(s) for s in sys.argv[3].split(",")]; njobs = int(sys.argv[4]); maxf = int(sys.argv[5]) if len(sys.argv) > 5 else 99
os.makedirs(outdir, exist_ok=True); X, y = build_features(); done = 0
for seed in seeds:
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
        f = f"{outdir}/s{seed}_f{k}.npz"
        if os.path.exists(f): continue
        if done >= maxf: sys.exit()
        t0 = time.time(); ytr = y[tr]; inner = list(StratifiedKFold(3, shuffle=True, random_state=900 + seed % 100 + k).split(X.iloc[tr], ytr))
        names = MEMBERS if kind == "members" else (BENCH_EXTRA if kind == "extra" else ["svm"])
        mk = (lambda n, Xa, ya: build_all(Xa, ya, njobs)[n]) if kind == "members" else (lambda n, Xa, ya: build_bench_extra(n))
        O = np.zeros((len(tr), len(names))); T = np.zeros((len(te), len(names)))
        for j, n in enumerate(names):
            for a, b in inner: O[b, j] = score_of(mk(n, X.iloc[tr[a]], ytr[a]).fit(X.iloc[tr[a]], ytr[a]), X.iloc[tr[b]])
            T[:, j] = score_of(mk(n, X.iloc[tr], ytr).fit(X.iloc[tr], ytr), X.iloc[te])
        np.savez(f, O=O, T=T, ytr=ytr, yte=y[te], pay_tr=X["PAY_0"].values[tr], pay_te=X["PAY_0"].values[te], models=np.array(names)); done += 1
        print(f"{kind} seed {seed} fold {k} en {time.time()-t0:.0f}s", flush=True)
