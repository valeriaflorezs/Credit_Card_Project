"""Estudios restantes de la evaluación ampliada (Entregable 3, Tabla 4).

Todos usan los mismos 5 pliegues externos (semilla 42) y, cuando hay que entrenar, los hiperparámetros que
la mejor combinación de cada modelo eligió en cada pliegue (``comparacion_final/mejor_por_modelo.csv``).
No hay búsqueda nueva de hiperparámetros, así que ningún estudio toca el diseño del modelo.

- conformal:     conjuntos de predicción conformes (Mondrian por clase, alfa = 0.10) sobre las probabilidades
                 fuera de pliegue. Para el pliegue k se calibra con los otros 4 y se mide la cobertura en k.
- recalibracion: regresión isotónica ajustada con los otros 4 pliegues y aplicada al pliegue k.
- errores:       F1 y tasa de error por subgrupos (estado de pago de septiembre, cupo, edad, educación, sexo).
- desplazamiento: entrenar en un grupo y probar en otro (edad <= 35 contra > 35; posgrado o universidad
                 contra secundaria u otros). Hiperparámetros del pliegue 0.
- semillas:      10 semillas del modelo con los pliegues fijos (50 ajustes por modelo).
- escalabilidad: tiempo de ajuste e inferencia al variar n (filas) y p (columnas), pliegue 0.
- cd:            diagrama de diferencia crítica (Nemenyi) de los 10 modelos con CS-FOF incluido.

Uso (desde la raíz): python experiments/robustez/ampliada.py [estudio ...]
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
RAIZ = Path(__file__).resolve().parents[2]
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ / "experiments" / "cs_fof_diseno")); sys.path.insert(0, str(AQUI))
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
from robustez import fabrica, ece, MODELOS

OOF = RAIZ / "experiments" / "comparacion_final" / "probabilidades_oof.npz"
FIG = RAIZ / "results" / "figures"
NOMBRE = {"cs_fof": "CS-FOF", "catboost": "CatBoost", "xgboost": "XGBoost", "lightgbm": "LightGBM",
          "random_forest": "Random Forest", "svm": "SVM", "knn": "KNN", "decision_tree": "Árbol de decisión",
          "logistic_regression": "Regresión logística", "naive_bayes": "Naive Bayes"}


def _datos():
    import datos
    cwd = os.getcwd(); os.chdir(RAIZ / "experiments" / "cs_fof_diseno")
    X, y, cols, folds = datos.cargar(); os.chdir(cwd)
    return X, y, cols, folds


def _params():
    best = pd.read_csv(RAIZ / "experiments" / "comparacion_final" / "mejor_por_modelo.csv").set_index("model")
    return {m: json.loads(best.loc[m, "best_params_per_fold"]) for m in MODELOS}


# ---------------------------------------------------------------- conformal
def conformal(alfa=0.10):
    """Predicción conforme Mondrian: el umbral se calibra por separado en cada clase, así la cobertura se
    garantiza en ambas (no solo en promedio, donde dominaría la clase mayoritaria)."""
    d = np.load(OOF); y = d["y"]; _, _, _, folds = _datos()
    filas = []
    for m in MODELOS:
        P = np.c_[1 - d[m], d[m]]
        for k, (_, te) in enumerate(folds):
            cal = np.setdiff1d(np.arange(len(y)), te)
            q = {}
            for c in (0, 1):
                s = 1 - P[cal][y[cal] == c, c]                       # puntaje de no conformidad
                n = len(s); q[c] = np.quantile(s, min(1, np.ceil((n + 1) * (1 - alfa)) / n), method="higher")
            S = np.c_[1 - P[te, 0] <= q[0], 1 - P[te, 1] <= q[1]]  # conjunto: clases cuyo puntaje cabe
            yt = y[te]; cubre = S[np.arange(len(te)), yt]
            filas.append(dict(model=m, fold=k, cobertura=cubre.mean(), cobertura_0=cubre[yt == 0].mean(),
                              cobertura_1=cubre[yt == 1].mean(), tamano_medio=S.sum(1).mean(),
                              singleton=(S.sum(1) == 1).mean(), vacio=(S.sum(1) == 0).mean()))
    T = pd.DataFrame(filas); T.to_csv(AQUI / "conformal.csv", index=False)
    R = T.groupby("model").agg(["mean", "std"]).drop(columns="fold").round(4); R.to_csv(AQUI / "conformal_resumen.csv")
    print(R[[(c, "mean") for c in ("cobertura", "cobertura_0", "cobertura_1", "tamano_medio", "singleton")]].to_string())


# ---------------------------------------------------------------- recalibración
def recalibracion():
    d = np.load(OOF); y = d["y"]; _, _, _, folds = _datos()
    filas = []
    for m in MODELOS:
        p = d[m]; q = np.empty_like(p)
        for _, te in folds:
            cal = np.setdiff1d(np.arange(len(y)), te)
            q[te] = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p[cal], y[cal]).predict(p[te])
        for etapa, v in (("original", p), ("isotonica", q)):
            filas.append(dict(model=m, etapa=etapa, brier=brier_score_loss(y, v), log_loss=log_loss(y, np.clip(v, 1e-6, 1 - 1e-6)),
                              ece=ece(y, v), roc_auc=roc_auc_score(y, v), prob_media=v.mean()))
        np.save(AQUI / f"_iso_{m}.npy", q)
    T = pd.DataFrame(filas); T.to_csv(AQUI / "recalibracion.csv", index=False); print(T.round(4).to_string(index=False))
    # curva de confiabilidad antes y después (CS-FOF y CatBoost)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    bins = np.linspace(0, 1, 11)
    for a, etapa in zip(ax, ("original", "isotonica")):
        a.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="Calibración perfecta")
        for m, col in (("cs_fof", "#c8a24a"), ("catboost", "#3b6ea5")):
            v = d[m] if etapa == "original" else np.load(AQUI / f"_iso_{m}.npy")
            b = np.clip(np.digitize(v, bins) - 1, 0, 9); ok = [i for i in range(10) if (b == i).sum() > 30]
            a.plot([v[b == i].mean() for i in ok], [y[b == i].mean() for i in ok], "o-", c=col, label=NOMBRE[m])
        a.set_title("Probabilidades originales" if etapa == "original" else "Tras regresión isotónica")
        a.set_xlabel("Probabilidad predicha de impago"); a.grid(alpha=.3)
    ax[0].set_ylabel("Frecuencia observada de impago"); ax[1].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "recalibracion_isotonica.png", dpi=150); plt.close(fig)
    for m in MODELOS: (AQUI / f"_iso_{m}.npy").unlink()


# ---------------------------------------------------------------- análisis de errores
def errores():
    d = np.load(OOF); y = d["y"]; X, _, cols, _ = _datos(); c = {n: i for i, n in enumerate(cols)}
    pay0 = X[:, c["PAY_0"]]; edad = X[:, c["AGE"]]; cupo = X[:, c["LIMIT_BAL_LOG"]]
    q = np.quantile(cupo, [.25, .5, .75])
    grupos = {
        "Pago sept.: al día o adelantado (PAY_0 ≤ 0)": pay0 <= 0,
        "Pago sept.: 1 mes de atraso": pay0 == 1,
        "Pago sept.: 2+ meses de atraso": pay0 >= 2,
        "Cupo: cuartil 1 (más bajo)": cupo <= q[0],
        "Cupo: cuartil 4 (más alto)": cupo > q[2],
        "Edad ≤ 30": edad <= 30, "Edad 31-45": (edad > 30) & (edad <= 45), "Edad > 45": edad > 45,
        "Educación: posgrado": X[:, c["EDUCATION_1"]] == 1, "Educación: universidad": X[:, c["EDUCATION_2"]] == 1,
        "Educación: secundaria": X[:, c["EDUCATION_3"]] == 1,
        "Sexo: hombre": X[:, c["SEX_2"]] == 0, "Sexo: mujer": X[:, c["SEX_2"]] == 1,
    }
    filas = []
    for g, msk in grupos.items():
        fila = dict(grupo=g, n=int(msk.sum()), tasa_impago=y[msk].mean())
        for m in MODELOS:
            pr = d[m][msk] > 0.5
            fila[f"f1_{m}"] = f1_score(y[msk], pr, zero_division=0)
            fila[f"fp_{m}"] = ((pr == 1) & (y[msk] == 0)).mean(); fila[f"fn_{m}"] = ((pr == 0) & (y[msk] == 1)).mean()
        filas.append(fila)
    T = pd.DataFrame(filas); T["cs_fof_menos_catboost"] = T["f1_cs_fof"] - T["f1_catboost"]
    T.to_csv(AQUI / "errores_subgrupos.csv", index=False)
    print(T[["grupo", "n", "tasa_impago", "f1_cs_fof", "f1_catboost", "cs_fof_menos_catboost", "fp_cs_fof", "fn_cs_fof"]].round(3).to_string(index=False))


# ---------------------------------------------------------------- cambio de distribución
def desplazamiento():
    X, y, cols, _ = _datos(); c = {n: i for i, n in enumerate(cols)}; P = _params()
    edad = X[:, c["AGE"]]; alta = (X[:, c["EDUCATION_1"]] == 1) | (X[:, c["EDUCATION_2"]] == 1)
    escenarios = {"Edad: entrena ≤ 35, prueba > 35": (edad <= 35, edad > 35),
                  "Educación: entrena universidad/posgrado, prueba secundaria/otros": (alta, ~alta)}
    filas = []
    for nom, (tr, te) in escenarios.items():
        tr, te = np.flatnonzero(tr), np.flatnonzero(te)
        for m in MODELOS:
            t = time.time(); mod = fabrica(m, P[m][0], y[tr]).fit(X[tr], y[tr]); p = mod.predict_proba(X[te])[:, 1]
            filas.append(dict(escenario=nom, model=m, n_train=len(tr), n_test=len(te), tasa_train=y[tr].mean(), tasa_test=y[te].mean(),
                              f1=f1_score(y[te], p > 0.5), roc_auc=roc_auc_score(y[te], p), brier=brier_score_loss(y[te], p),
                              seconds=round(time.time() - t, 1)))
            print(nom, m, round(filas[-1]["f1"], 4), flush=True)
    T = pd.DataFrame(filas); T.to_csv(AQUI / "desplazamiento.csv", index=False); print(T.round(4).to_string(index=False))


# ---------------------------------------------------------------- semillas
def semillas(n=10):
    X, y, _, folds = _datos(); P = _params(); salida = AQUI / "semillas.csv"
    hechos = pd.read_csv(salida) if salida.exists() else pd.DataFrame(columns=["model", "semilla", "fold"])
    for s in range(n):
        for m in MODELOS:
            for k, (tr, te) in enumerate(folds):
                if ((hechos["model"] == m) & (hechos["semilla"] == s) & (hechos["fold"] == k)).any():
                    continue
                mod = fabrica(m, P[m][k], y[tr])
                mod.set_params(**({"random_seed": s} if m == "catboost" else {"random_state": s}))
                p = mod.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                fila = dict(model=m, semilla=s, fold=k, f1=f1_score(y[te], p > 0.5), roc_auc=roc_auc_score(y[te], p))
                hechos = pd.concat([hechos, pd.DataFrame([fila])], ignore_index=True); hechos.to_csv(salida, index=False)
            print(f"semilla {s} {m:14s} F1 {hechos[(hechos.model == m) & (hechos.semilla == s)].f1.mean():.4f}", flush=True)
    resumen_semillas()


def resumen_semillas():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    h = pd.read_csv(AQUI / "semillas.csv"); m_s = h.groupby(["model", "semilla"]).f1.mean().reset_index()
    R = m_s.groupby("model").f1.agg(["mean", "std", "min", "max"])
    R["ic95_bajo"] = m_s.groupby("model").f1.quantile(0.025); R["ic95_alto"] = m_s.groupby("model").f1.quantile(0.975)
    # cuántas semillas gana CS-FOF a cada modelo (mismo pliegue y semilla)
    piv = m_s.pivot(index="semilla", columns="model", values="f1")
    R["cs_fof_gana_en"] = [int((piv["cs_fof"] > piv[m]).sum()) if m != "cs_fof" else np.nan for m in R.index]
    R = R.sort_values("mean", ascending=False).round(4); R.to_csv(AQUI / "semillas_resumen.csv"); print(R.to_string())
    orden = list(R.index)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([h[h.model == m].f1 for m in orden], tick_labels=[NOMBRE[m] for m in orden], showmeans=True)
    for i, m in enumerate(orden, 1):
        ax.scatter(np.full(10, i) + np.random.default_rng(i).uniform(-.12, .12, 10), m_s[m_s.model == m].f1, s=12, c="#c8a24a", zorder=3)
    ax.set_ylabel("F1 de la clase de impago"); ax.set_title("F1 en 10 semillas × 5 pliegues externos (puntos: media por semilla)")
    ax.grid(alpha=.3, axis="y"); fig.tight_layout(); fig.savefig(FIG / "semillas_boxplot.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- escalabilidad
def escalabilidad():
    X, y, cols, folds = _datos(); P = _params(); tr, te = folds[0]; rng = np.random.default_rng(0)
    filas = []
    for eje, niveles in (("n", (0.10, 0.25, 0.50, 1.0)), ("p", (0.25, 0.50, 0.75, 1.0))):
        for f in niveles:
            idx = rng.choice(tr, int(f * len(tr)), replace=False) if eje == "n" else tr
            colz = np.arange(X.shape[1]) if eje == "p" and f == 1 else (np.sort(rng.choice(X.shape[1], max(2, int(round(f * X.shape[1]))), replace=False)) if eje == "p" else np.arange(X.shape[1]))
            for m in MODELOS:
                mod = fabrica(m, P[m][0], y[idx]); t = time.perf_counter(); mod.fit(X[idx][:, colz], y[idx]); ta = time.perf_counter() - t
                t = time.perf_counter(); mod.predict_proba(X[te][:, colz]); ti = time.perf_counter() - t
                filas.append(dict(eje=eje, fraccion=f, n=len(idx), p=len(colz), model=m, ajuste_s=round(ta, 2),
                                  inferencia_ms_por_1000=round(1000 * ti / len(te) * 1000, 3)))
                print(filas[-1], flush=True)
    T = pd.DataFrame(filas); T.to_csv(AQUI / "escalabilidad.csv", index=False)
    # pendiente log-log del tiempo de ajuste de cada modelo frente a n
    for m in MODELOS:
        s = T[(T.eje == "n") & (T.model == m)]; b = np.polyfit(np.log(s.n), np.log(s.ajuste_s), 1)[0]
        print(f"{m:14s} tiempo ∝ n^{b:.2f}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m in MODELOS:
        for a, eje in zip(ax, ("n", "p")):
            s = T[(T.eje == eje) & (T.model == m)]; a.plot(s[eje], s.ajuste_s, "o-", label=NOMBRE[m])
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("Filas de entrenamiento (n, escala log)")
    ax[1].set_xlabel("Columnas (p)"); ax[0].set_ylabel("Tiempo de ajuste (s)"); ax[1].legend(fontsize=8)
    for a in ax: a.grid(alpha=.3)
    fig.suptitle("Tiempo de ajuste al variar n y p (pliegue 0, 4 núcleos)"); fig.tight_layout()
    fig.savefig(FIG / "escalabilidad.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- diagrama de diferencia crítica
def cd():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    J = json.load(open(RAIZ / "experiments" / "comparacion_final" / "friedman.json"))
    r = pd.Series(J["rangos"]).sort_values(); cdv = J["nemenyi_cd"]; k = len(r)
    fig, ax = plt.subplots(figsize=(9, 3.6)); ax.set_xlim(0.5, k + .5); ax.set_ylim(0, 1); ax.axis("off")
    ax.hlines(0.8, 1, k, color="k")
    for i in range(1, k + 1):
        ax.vlines(i, .78, .82, color="k"); ax.text(i, .84, str(i), ha="center")
    ax.hlines(0.95, 1, 1 + cdv, color="#b03a2e", lw=2); ax.text(1 + cdv / 2, .97, f"DC = {cdv:.2f}", ha="center", color="#b03a2e")
    for j, (m, v) in enumerate(r.items()):
        izq = j < k / 2; yy = 0.62 - 0.11 * (j if izq else k - 1 - j)
        ax.plot([v, v], [0.8, yy], c="k", lw=.8); x2 = 0.6 if izq else k + .4
        ax.plot([v, x2], [yy, yy], c="k", lw=.8)
        ax.text(x2 + (-.05 if izq else .05), yy, f"{NOMBRE[m]} ({v:.1f})", ha="right" if izq else "left", va="center",
                fontweight="bold" if m == "cs_fof" else None, color="#a07818" if m == "cs_fof" else "k")
    # barras de grupos no distinguibles
    grupos, i = [], 0
    vals = r.values
    for a in range(k):
        b = max(j for j in range(k) if vals[j] - vals[a] <= cdv)
        if b > a and (not grupos or b > grupos[-1][1]): grupos.append((a, b))
    for n_g, (a, b) in enumerate(grupos):
        ax.hlines(0.72 - 0.035 * n_g, vals[a] - .05, vals[b] + .05, color="#3b6ea5", lw=4, alpha=.8)
    ax.set_title(f"Diferencia crítica de Nemenyi (10 modelos, 5 pliegues; Friedman χ² = {J['friedman_chi2']:.1f}, p = {J['friedman_p']:.1e})", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "cd_diagram_final.png", dpi=150); plt.close(fig); print(r.to_string(), "\nDC", cdv, grupos)


if __name__ == "__main__":
    for e in sys.argv[1:] or ["cd", "conformal", "recalibracion", "errores", "desplazamiento", "escalabilidad", "semillas"]:
        globals()[e]()
