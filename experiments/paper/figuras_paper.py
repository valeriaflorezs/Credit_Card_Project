"""Figuras del paper (sección de resultados): PDF vectorial, texto en inglés, fuente sin serifa incrustada.

Uso (desde la raíz, después de resultados_paper.py): python experiments/paper/figuras_paper.py
Salida: experiments/paper/figs/*.pdf
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, average_precision_score  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
FIGS = AQUI / "figs"; FIGS.mkdir(exist_ok=True)
CF = RAIZ / "experiments" / "comparacion_final"
RB = RAIZ / "experiments" / "robustez"
sys.path.insert(0, str(AQUI))
from resultados_paper import NOMBRE  # noqa: E402

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "pdf.fonttype": 42,
                     "ps.fonttype": 42, "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
GOLD, NAVY, GREY = "#B88F3D", "#36456D", "#8089A0"
EN = {k: v.replace(" (proposed)", "") for k, v in NOMBRE.items()}


def guardar(fig, nombre):
    fig.savefig(FIGS / f"{nombre}.pdf", bbox_inches="tight"); plt.close(fig)


def f1_por_pliegue():
    T = pd.read_csv(AQUI / "metricas_pliegues.csv")
    orden = T.groupby("model").f1.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    for i, m in enumerate(orden):
        v = T[T.model == m].f1.to_numpy(); col = GOLD if m == "cs_fof" else NAVY
        ax.boxplot(v, positions=[i], vert=False, widths=0.55, patch_artist=True,
                   boxprops=dict(facecolor=col, alpha=0.25, edgecolor=col), medianprops=dict(color=col),
                   whiskerprops=dict(color=col), capprops=dict(color=col), flierprops=dict(markeredgecolor=col))
        ax.scatter(v, np.full(5, i), s=10, color=col, zorder=3)
    ax.set_yticks(range(len(orden)), [EN[m] for m in orden]); ax.grid(axis="x", alpha=.3)
    ax.set_xlabel("F1 of the default class (5 outer folds, threshold 0.5)")
    guardar(fig, "fig_f1_folds")


def intervalos():
    R = pd.read_csv(AQUI / "pruebas_cs_fof.csv").sort_values("dif_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    y = np.arange(len(R))
    for i, r in enumerate(R.itertuples()):
        col = NAVY if r.p_nb_holm < 0.05 else GREY
        ax.plot([r.ic95_bajo, r.ic95_alto], [i, i], color=col, lw=2)
        ax.scatter(r.dif_f1, i, color=col, zorder=3, s=22)
    ax.axvline(0, color="k", lw=.8); ax.axvspan(-0.01, 0.01, color=GOLD, alpha=.15, label="ROPE (±0.01)")
    ax.set_yticks(y, [EN[m] for m in R.baseline]); ax.invert_yaxis(); ax.grid(axis="x", alpha=.3)
    ax.set_xlabel("F1 difference, CS-FOF minus baseline (mean and 95% CI, corrected t)")
    ax.legend(loc="lower right"); guardar(fig, "fig_diff_intervals")


def roc_pr_cal_cm():
    d = np.load(CF / "probabilidades_oof.npz"); y = d["y"]
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for m, col in (("cs_fof", GOLD), ("catboost", NAVY)):
        fpr, tpr, _ = roc_curve(y, d[m]); ax[0].plot(fpr, tpr, color=col, label=f"{EN[m]} (AUC {roc_auc_score(y, d[m]):.3f})")
        pr, rc, _ = precision_recall_curve(y, d[m]); ax[1].plot(rc, pr, color=col, label=f"{EN[m]} (AP {average_precision_score(y, d[m]):.3f})")
        fp, mp = calibration_curve(y, d[m], n_bins=10); ax[2].plot(mp, fp, "o-", ms=3, color=col, label=EN[m])
    ax[0].plot([0, 1], [0, 1], ls="--", color=GREY, lw=.8); ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    ax[1].axhline(y.mean(), ls="--", color=GREY, lw=.8); ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    ax[2].plot([0, 1], [0, 1], ls="--", color=GREY, lw=.8); ax[2].set(xlabel="Mean predicted probability", ylabel="Observed default rate", title="Calibration (10 bins)")
    for a in ax: a.grid(alpha=.3); a.legend(loc="lower right" if a is not ax[1] else "upper right")
    fig.tight_layout(); guardar(fig, "fig_roc_pr_calibration")
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.9))
    for a, m in zip(ax, ("cs_fof", "catboost")):
        cm = confusion_matrix(y, d[m] >= 0.5)
        a.imshow(cm, cmap="Blues"); a.set_title(f"{EN[m]} (threshold 0.5)")
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        a.set_xticks([0, 1], ["Pays", "Defaults"]); a.set_yticks([0, 1], ["Pays", "Defaults"])
        a.set_xlabel("Predicted"); a.set_ylabel("Actual")
    fig.tight_layout(); guardar(fig, "fig_confusion")


def ablacion():
    A = pd.read_csv(AQUI / "ablacion_pliegues.csv")
    nombres = {"completo": "Full CS-FOF", "sin_oblicuo": "Without oblique cut", "sin_costo": "Without cost weighting",
               "todas_las_variables": "Direction on all node variables"}
    orden = ["completo", "sin_oblicuo", "todas_las_variables", "sin_costo"]
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.8))
    for a, met, lab in ((ax[0], "f1", "F1 (default class)"), (ax[1], "roc_auc", "ROC-AUC")):
        g = A.groupby("variante")[met].agg(["mean", "std"]).reindex(orden)
        a.barh(range(4), g["mean"], xerr=g["std"], color=[GOLD] + [NAVY] * 3, alpha=.8, capsize=3)
        a.set_yticks(range(4), [nombres[v] for v in orden] if a is ax[0] else [""] * 4); a.invert_yaxis()
        lo = (g["mean"] - g["std"]).min(); a.set_xlim(lo - 0.01, (g["mean"] + g["std"]).max() + 0.005)
        a.set_xlabel(f"{lab}, mean ± SD over 5 outer folds"); a.grid(axis="x", alpha=.3)
    fig.tight_layout(); guardar(fig, "fig_ablation")


def sensibilidad():
    ruido = pd.read_csv(RB / "ruido.csv"); tam = pd.read_csv(RB / "tamano.csv"); sem = pd.read_csv(RB / "semillas.csv")
    mods = ["cs_fof", "catboost", "xgboost", "lightgbm", "random_forest"]
    cols = {"cs_fof": GOLD, "catboost": NAVY, "xgboost": "#5B8DB8", "lightgbm": "#7A9E7E", "random_forest": GREY}
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for m in mods:
        g = tam[tam.model == m].groupby("nivel").f1.mean(); ax[0].plot(g.index * 100, g.values, "o-", ms=3, color=cols[m], label=EN[m])
        g = ruido[ruido.model == m].groupby("nivel").f1.mean(); ax[1].plot(g.index * 100, g.values, "o-", ms=3, color=cols[m])
    ax[0].set(xlabel="Training data used (%)", ylabel="F1 (mean of 5 folds)", title="Learning curve"); ax[0].set_xscale("log")
    ax[0].set_xticks([10, 25, 50, 100], ["10", "25", "50", "100"])
    ax[1].set(xlabel="Training labels flipped (%)", title="Label noise")
    ms = sem.groupby(["model", "semilla"]).f1.mean().reset_index()
    ax[2].boxplot([ms[ms.model == m].f1 for m in mods], tick_labels=[EN[m] for m in mods], vert=True)
    ax[2].set(ylabel="F1 (mean over folds, per seed)", title="10 random seeds"); ax[2].tick_params(axis="x", rotation=35)
    for a in ax: a.grid(alpha=.3)
    ax[0].legend(fontsize=7); fig.tight_layout(); guardar(fig, "fig_sensitivity")


def shap():
    W = pd.read_csv(RB / "shap_cs_fof_catboost.csv", index_col=0).head(10)[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 3.4)); yy = np.arange(len(W))
    ax.barh(yy + .2, W.cs_fof, .4, color=GOLD, label="CS-FOF"); ax.barh(yy - .2, W.catboost, .4, color=NAVY, label="CatBoost")
    ax.set_yticks(yy, W.index); ax.set_xlabel("Mean |SHAP value| (probability of default)"); ax.legend(); ax.grid(axis="x", alpha=.3)
    guardar(fig, "fig_shap")


if __name__ == "__main__":
    for f in (f1_por_pliegue, intervalos, roc_pr_cal_cm, ablacion, sensibilidad, shap):
        f(); print("ok", f.__name__)
