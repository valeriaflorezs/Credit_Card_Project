"""Figuras del capítulo de evaluación ampliada, redibujadas con CS-FOF y los dos mejores modelos de referencia.

No reentrena nada: lee semillas.csv, escalabilidad.csv y las probabilidades fuera de pliegue.
Uso (desde la raíz): python experiments/robustez/figuras_libro.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from ampliada import FIG, NOMBRE, OOF, _datos  # noqa: E402

MODELOS = ["cs_fof", "xgboost", "random_forest"]
COLOR = {"cs_fof": "#c8a24a", "xgboost": "#3b6ea5", "random_forest": "#4f7f58"}


def semillas():
    h = pd.read_csv(AQUI / "semillas.csv"); h = h[h.model.isin(MODELOS)]
    m_s = h.groupby(["model", "semilla"]).f1.mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([h[h.model == m].f1 for m in MODELOS], tick_labels=[NOMBRE[m] for m in MODELOS], showmeans=True)
    for i, m in enumerate(MODELOS, 1):
        ax.scatter(np.full(10, i) + np.random.default_rng(i).uniform(-.12, .12, 10), m_s[m_s.model == m].f1, s=12, c="#c8a24a", zorder=3)
    ax.set_ylabel("F1 de la clase de impago"); ax.set_title("F1 en 10 semillas × 5 pliegues externos (puntos: media por semilla)")
    ax.grid(alpha=.3, axis="y"); fig.tight_layout(); fig.savefig(FIG / "semillas_boxplot.png", dpi=150); plt.close(fig)


def escalabilidad():
    T = pd.read_csv(AQUI / "escalabilidad.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m in MODELOS:
        for a, eje in zip(ax, ("n", "p")):
            s = T[(T.eje == eje) & (T.model == m)]; a.plot(s[eje], s.ajuste_s, "o-", label=NOMBRE[m], c=COLOR[m])
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("Filas de entrenamiento (n, escala log)")
    ax[1].set_xlabel("Columnas (p)"); ax[0].set_ylabel("Tiempo de ajuste (s)"); ax[1].legend(fontsize=8)
    for a in ax: a.grid(alpha=.3)
    fig.suptitle("Tiempo de ajuste al variar n y p (pliegue 0, 4 núcleos)"); fig.tight_layout()
    fig.savefig(FIG / "escalabilidad.png", dpi=150); plt.close(fig)
    for m in MODELOS:
        s = T[(T.eje == "n") & (T.model == m)]; print(f"{m:14s} tiempo ∝ n^{np.polyfit(np.log(s.n), np.log(s.ajuste_s), 1)[0]:.2f}")


def recalibracion():
    d = np.load(OOF); y = d["y"]; _, _, _, folds = _datos()
    iso = {}
    for m in ("cs_fof", "xgboost"):
        p = d[m]; q = np.empty_like(p)
        for _, te in folds:
            cal = np.setdiff1d(np.arange(len(y)), te)
            q[te] = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p[cal], y[cal]).predict(p[te])
        iso[m] = q
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True); bins = np.linspace(0, 1, 11)
    for a, etapa in zip(ax, ("original", "isotonica")):
        a.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="Calibración perfecta")
        for m in ("cs_fof", "xgboost"):
            v = d[m] if etapa == "original" else iso[m]
            b = np.clip(np.digitize(v, bins) - 1, 0, 9); ok = [i for i in range(10) if (b == i).sum() > 30]
            a.plot([v[b == i].mean() for i in ok], [y[b == i].mean() for i in ok], "o-", c=COLOR[m], label=NOMBRE[m])
        a.set_title("Probabilidades originales" if etapa == "original" else "Tras regresión isotónica")
        a.set_xlabel("Probabilidad predicha de impago"); a.grid(alpha=.3)
    ax[0].set_ylabel("Frecuencia observada de impago"); ax[1].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "recalibracion_isotonica.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    semillas(); escalabilidad(); recalibracion(); print("ok")
