"""SHAP del modelo original (CS-FOF) frente al mejor benchmark (CatBoost). Entregable 3, sección 5.4.

- Mismo método para los dos: ``shap.PermutationExplainer`` (agnóstico al modelo), porque CS-FOF no es un
  árbol de scikit-learn y TreeExplainer no aplica. Usar el mismo explicador evita que las diferencias vengan
  del método y no del modelo.
- Pliegue externo 1 (índice 0) de la validación anidada: cada modelo se entrena con su entrenamiento y los
  hiperparámetros que eligió en ese pliegue; se explican 300 clientes de su prueba (muestra estratificada,
  semilla 42) que el modelo no vio.
- Fondo (distribución de referencia): 50 clientes del entrenamiento, semilla 42. 101 evaluaciones por cliente
  (dos permutaciones antitéticas de las 25 columnas).
- Se explica la probabilidad de incumplimiento (clase 1).

Salidas: ``shap_cs_fof_catboost.csv`` (importancia media |SHAP| por variable y modelo),
``shap_valores.npz`` y la figura ``results/figures/shap_cs_fof_vs_catboost.png``.
Uso (desde la raíz): python experiments/robustez/shap_cs_fof.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(AQUI))
from ampliada import _datos, _params, FIG  # noqa: E402
from robustez import fabrica  # noqa: E402

MODELOS = ("cs_fof", "catboost")
NOMBRE = {"cs_fof": "CS-FOF", "catboost": "CatBoost"}


def main():
    X, y, cols, folds = _datos()
    P = _params()
    tr, te = folds[0]
    rng = np.random.default_rng(42)
    idx = np.concatenate([rng.choice(te[y[te] == c], n, replace=False) for c, n in ((0, 234), (1, 66))])
    fondo = X[rng.choice(tr, 50, replace=False)]
    valores, filas = {}, []
    for m in MODELOS:
        t = time.time()
        modelo = fabrica(m, P[m][0], y[tr]).fit(X[tr], y[tr])
        f = lambda Z, mod=modelo: mod.predict_proba(Z)[:, 1]
        exp = shap.PermutationExplainer(f, shap.maskers.Independent(fondo), seed=42)
        sv = exp(X[idx], max_evals=2 * X.shape[1] + 51, silent=True).values
        valores[m] = sv
        aditividad = float(np.max(np.abs(sv.sum(1) + f(fondo).mean() - f(X[idx]))))
        print(f"{m}: {time.time() - t:.0f} s, error de aditividad máx {aditividad:.2e}", flush=True)
        for j, c in enumerate(cols):
            filas.append(dict(model=m, variable=c, media_abs=float(np.abs(sv[:, j]).mean()),
                              media=float(sv[:, j].mean())))
    T = pd.DataFrame(filas)
    W = T.pivot(index="variable", columns="model", values="media_abs").sort_values("catboost", ascending=False)
    W.to_csv(AQUI / "shap_cs_fof_catboost.csv")
    np.savez(AQUI / "shap_valores.npz", X=X[idx], y=y[idx], columnas=np.array(cols), **valores)

    rho, p = spearmanr(W["cs_fof"], W["catboost"])
    top = {m: list(W[m].sort_values(ascending=False).index[:5]) for m in MODELOS}
    comunes = sorted(set(top["cs_fof"]) & set(top["catboost"]))
    # concordancia de signo por cliente en las 5 variables principales de cada modelo
    j5 = [cols.index(c) for c in set(top["cs_fof"]) | set(top["catboost"])]
    signo = float((np.sign(valores["cs_fof"][:, j5]) == np.sign(valores["catboost"][:, j5])).mean())
    res = dict(spearman_importancias=round(float(rho), 3), p=float(p), top5=top, top5_comunes=comunes,
               concordancia_signo_top=round(signo, 3))
    (AQUI / "shap_resumen.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(W.round(4).to_string()); print(json.dumps(res, ensure_ascii=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    orden = list(W.index[:12])[::-1]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    yy = np.arange(len(orden))
    ax[0].barh(yy + 0.2, W.loc[orden, "cs_fof"], 0.4, color="#B88F3D", label="CS-FOF")
    ax[0].barh(yy - 0.2, W.loc[orden, "catboost"], 0.4, color="#36456D", label="CatBoost")
    ax[0].set_yticks(yy, orden, fontsize=9); ax[0].set_xlabel("Importancia media |SHAP| (probabilidad)")
    ax[0].legend(loc="lower right"); ax[0].set_title(f"12 variables más importantes (Spearman ρ = {rho:.2f})", fontsize=10)
    for a, m in zip(ax[1:], MODELOS):
        sv = valores[m]; js = [cols.index(c) for c in orden]
        for k, j in enumerate(js):
            v = X[idx][:, j]; z = (v - v.min()) / (np.ptp(v) or 1)
            a.scatter(sv[:, j], np.full(len(v), k) + rng.uniform(-.25, .25, len(v)), c=z, cmap="coolwarm", s=6, alpha=.7)
        a.axvline(0, color="grey", lw=.8); a.set_yticks(yy, orden if m == "cs_fof" else [], fontsize=9)
        a.set_title(f"{NOMBRE[m]}: efecto por cliente (rojo = valor alto)", fontsize=10); a.set_xlabel("Valor SHAP")
    fig.suptitle("SHAP por permutación en 300 clientes del pliegue externo 1 (mismo explicador y fondo para los dos)")
    fig.tight_layout(); fig.savefig(FIG / "shap_cs_fof_vs_catboost.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
