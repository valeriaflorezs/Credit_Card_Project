"""Estudios de robustez en Taiwán (Entregable 3, sección 6.5): ruido en las etiquetas, tamaño de muestra
y calibración de probabilidades.

Para que el costo sea bajo y no haya búsqueda nueva, cada modelo usa los hiperparámetros que su mejor
combinación eligió en cada pliegue externo (``experiments/comparacion_final/mejor_por_modelo.csv``).
Las particiones son las mismas 5 externas (semilla 42). El ruido y el submuestreo se aplican SOLO al
conjunto de entrenamiento de cada pliegue; la prueba siempre es limpia y completa.

- Ruido: se invierte una fracción r de las etiquetas de entrenamiento (r = 0, 5, 10, 20 %).
- Tamaño: se entrena con una fracción estratificada f del entrenamiento (f = 10, 25, 50, 100 %).
- Calibración: Brier, log-loss y ECE (10 bins) de las probabilidades fuera de pliegue ya calculadas.

Uso (desde la raíz): python experiments/robustez/robustez.py [ruido|tamano|calibracion]
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ / "experiments" / "cs_fof_diseno"))
AQUI = Path(__file__).resolve().parent
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from src.original_model import CSFisherObliqueForest

MODELOS = ("cs_fof", "catboost", "xgboost", "lightgbm", "random_forest")
RUIDO = (0.0, 0.05, 0.10, 0.20)
TAMANO = (0.10, 0.25, 0.50, 1.0)


def fabrica(modelo, params, yt):
    spw = (yt == 0).sum() / (yt == 1).sum()
    if modelo == "cs_fof":
        return CSFisherObliqueForest(class_weight="balanced", random_state=42, n_jobs=4, **params)
    if modelo == "xgboost":
        return XGBClassifier(random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=4, scale_pos_weight=spw, **params)
    if modelo == "random_forest":
        return RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=4, **params)
    if modelo == "lightgbm":
        return LGBMClassifier(class_weight="balanced", random_state=42, n_jobs=4, subsample_freq=1, verbose=-1, **params)
    return CatBoostClassifier(auto_class_weights="Balanced", random_seed=42, thread_count=4, verbose=0,
                              allow_writing_files=False, **params)


def perturbar(estudio, nivel, y_tr, rng):
    """Devuelve (índices de entrenamiento a usar, etiquetas posiblemente ruidosas)."""
    idx = np.arange(len(y_tr)); yy = y_tr.copy()
    if estudio == "ruido" and nivel > 0:
        voltear = rng.choice(len(yy), int(round(nivel * len(yy))), replace=False)
        yy[voltear] = 1 - yy[voltear]
    if estudio == "tamano" and nivel < 1:
        idx = np.concatenate([rng.choice(np.flatnonzero(y_tr == c), int(round(nivel * (y_tr == c).sum())), replace=False)
                              for c in (0, 1)])
    return idx, yy


def correr(estudio):
    import datos
    os.chdir(RAIZ / "experiments" / "cs_fof_diseno")
    X, y, _, folds = datos.cargar()
    best = pd.read_csv(RAIZ / "experiments" / "comparacion_final" / "mejor_por_modelo.csv").set_index("model")
    niveles = RUIDO if estudio == "ruido" else TAMANO
    salida = AQUI / f"{estudio}.csv"
    hechos = pd.read_csv(salida) if salida.exists() else pd.DataFrame(columns=["model", "nivel", "fold"])
    for nivel in niveles:
        for modelo in MODELOS:
            P = json.loads(best.loc[modelo, "best_params_per_fold"])
            for k, (tr, te) in enumerate(folds):
                if ((hechos["model"] == modelo) & np.isclose(hechos["nivel"].astype(float), nivel) & (hechos["fold"] == k)).any():
                    continue
                rng = np.random.default_rng(1000 * k + int(round(100 * nivel)))  # mismo ruido para todos los modelos
                idx, yy = perturbar(estudio, nivel, y[tr], rng)
                t = time.time()
                m = fabrica(modelo, P[k], yy[idx]).fit(X[tr][idx], yy[idx])
                p = m.predict_proba(X[te])[:, 1]
                fila = dict(estudio=estudio, model=modelo, nivel=nivel, fold=k, n_train=len(idx),
                            f1=f1_score(y[te], p > 0.5), roc_auc=roc_auc_score(y[te], p), seconds=round(time.time() - t, 1))
                hechos = pd.concat([hechos, pd.DataFrame([fila])], ignore_index=True); hechos.to_csv(salida, index=False)
                print(f"{estudio} {nivel:.2f} {modelo:14s} pliegue {k + 1}: F1 {fila['f1']:.4f} AUC {fila['roc_auc']:.4f} ({fila['seconds']:.0f}s)", flush=True)
    resumen = hechos.groupby(["model", "nivel"])[["f1", "roc_auc"]].agg(["mean", "std"]).round(4)
    print(resumen.to_string()); resumen.to_csv(AQUI / f"{estudio}_resumen.csv")


def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(p[b == i].mean() - y[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any())


def calibracion():
    d = np.load(RAIZ / "experiments" / "comparacion_final" / "probabilidades_oof.npz"); y = d["y"]
    filas = [dict(model=m, brier=brier_score_loss(y, d[m]), log_loss=log_loss(y, np.clip(d[m], 1e-6, 1 - 1e-6)),
                  ece=ece(y, d[m]), prob_media=d[m].mean(), tasa_real=y.mean()) for m in MODELOS]
    T = pd.DataFrame(filas).sort_values("brier"); print(T.round(4).to_string(index=False)); T.to_csv(AQUI / "calibracion.csv", index=False)


if __name__ == "__main__":
    estudios = sys.argv[1:] or ["calibracion", "ruido", "tamano"]
    for e in estudios:
        calibracion() if e == "calibracion" else correr(e)
