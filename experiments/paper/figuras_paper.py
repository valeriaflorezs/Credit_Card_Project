"""Todas las figuras del paper, generadas desde un único script con las reglas de la guía de redacción.

Reglas: PDF vectorial; tamaño final de impresión (ancho útil de 6.3 pulgadas, se insertan con width=\\linewidth);
texto base de 12 pt (13 pt en títulos de ejes) en fuente sin serifa incrustada; rótulos en inglés; sin título dentro
de la figura (va en el pie); líneas de 2 pt; color combinado con trama, estilo de línea o marcador; leyenda sin marco.

Uso (desde la raíz, después de resultados_paper.py y apendice_eda.py): python experiments/paper/figuras_paper.py
Salida: experiments/paper/figures/*.pdf
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
FIGS = AQUI / "figures"; FIGS.mkdir(exist_ok=True)
CF = RAIZ / "experiments" / "comparacion_final"
RB = RAIZ / "experiments" / "robustez"
sys.path.insert(0, str(AQUI))
from apendice_eda import BILL, NUM, PAMT, PAY, Y, cargar  # noqa: E402,F401
from resultados_paper import NOMBRE  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["LM Sans 12", "Latin Modern Sans", "DejaVu Sans"],  # LM Sans 12 = Latin Modern Sans en su corte de 12 pt
    "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "lines.linewidth": 2.0, "axes.grid": True, "grid.alpha": 0.3,
    "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "standard",  # sin recorte: el PDF queda exactamente con el ancho útil de la página
})
ANCHO = 6.3
GOLD, NAVY, GREY = "#B88F3D", "#36456D", "#6F7890"
EN = {k: v.replace(" (proposed)", "") for k, v in NOMBRE.items()}
EN.update({"logistic_regression": "Logistic reg. (tuned)", "lr_none": "Logistic reg. (no penalty)",
           "ridge": "Ridge (L2 logistic)", "lasso": "Lasso (L1 logistic)"})
# Estilo por modelo: color + estilo de línea + marcador, para que se distingan también en escala de grises
ESTILO = {"cs_fof": (GOLD, "-", "o"), "catboost": (NAVY, "--", "s"), "xgboost": ("#5B8DB8", "-.", "^"),
          "lightgbm": ("#4F7F58", ":", "D"), "random_forest": (GREY, (0, (5, 1, 1, 1)), "v")}


def guardar(fig, nombre):
    fig.tight_layout(pad=0.3); fig.savefig(FIGS / f"{nombre}.pdf"); plt.close(fig)


def etiqueta(ax, txt, x=0.03, y=0.95, **kw):
    ax.text(x, y, txt, transform=ax.transAxes, va="top", ha="left", fontsize=12, **kw)


def leyenda_modelos(ax, mods, extra=()):
    h = [Line2D([], [], color=ESTILO[m][0], ls=ESTILO[m][1], marker=ESTILO[m][2], ms=6, label=EN[m]) for m in mods]
    ax.axis("off"); ax.legend(handles=h + list(extra), frameon=False, loc="center", handlelength=3.2)


# ------------------------------------------------------------------ EDA (metodología)
def eda():
    R = json.loads((AQUI / "apendice_eda.json").read_text(encoding="utf-8"))
    _, d = cargar()
    g0, g1 = d[d[Y] == 0], d[d[Y] == 1]

    v = [R["clase"]["0"], R["clase"]["1"]]
    fig, ax = plt.subplots(figsize=(ANCHO, 2.1))
    b = ax.barh(["No default (0)", "Default (1)"], v, color=[NAVY, GOLD], height=0.6)
    b[1].set_hatch("//"); b[1].set_edgecolor("white")
    for i, x in enumerate(v):
        ax.text(x + 300, i, f"{x:,} ({x / sum(v):.1%})", va="center")
    ax.set_xlabel("Number of clients"); ax.set_xlim(0, max(v) * 1.38); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    guardar(fig, "fig_eda_target")

    vals = [-2, -1, 0, 1, 2, 3, "4+"]; w = 0.4   # 4 a 8 meses se agrupan: son menos del 1 % de cada clase
    d0 = g0.PAY_0.clip(upper=4).value_counts(normalize=True).reindex(range(-2, 5), fill_value=0).to_numpy() * 100
    d1 = g1.PAY_0.clip(upper=4).value_counts(normalize=True).reindex(range(-2, 5), fill_value=0).to_numpy() * 100
    fig, ax = plt.subplots(figsize=(ANCHO, 3.5)); x = np.arange(len(vals))
    b0 = ax.bar(x - w / 2, d0, w, color=NAVY, label="No default")
    b1 = ax.bar(x + w / 2, d1, w, color=GOLD, hatch="//", edgecolor="white", label="Default")
    for bars in (b0, b1):
        for r in bars:
            if r.get_height() >= 0.5:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.8, f"{r.get_height():.0f}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x, vals); ax.set_xlabel("PAY_0: repayment status in September (months of delay)")
    ax.set_ylabel("Clients in each class (%)"); ax.set_ylim(0, max(d0.max(), d1.max()) * 1.17)
    ax.grid(axis="x", visible=False); ax.legend(frameon=False)
    guardar(fig, "fig_eda_pay0")

    lab = {"1": "Graduate school", "2": "University", "3": "High school", "4": "Others"}
    tasas = R["chi2"]["EDUCATION"]["tasas"]; ks = sorted(tasas)
    fig, ax = plt.subplots(figsize=(ANCHO, 3.2))
    bb = ax.bar([lab[k] for k in ks], [tasas[k] for k in ks], color=NAVY, width=0.6)
    for r in bb:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.004, f"{r.get_height():.1%}", ha="center", va="bottom")
    ax.axhline(R["clase"]["tasa"], ls="--", color=GOLD, label=f"Overall default rate ({R['clase']['tasa']:.1%})")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0)); ax.set_ylabel("Default rate (%)"); ax.set_xlabel("Education level")
    ax.set_ylim(0, 0.34); ax.grid(axis="x", visible=False); ax.legend(frameon=False, loc="upper right")
    guardar(fig, "fig_eda_rate_educ")

    sp = d[NUM].corr("spearman")
    fig, ax = plt.subplots(figsize=(ANCHO, 5.6))
    im = ax.imshow(sp.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(NUM)), NUM, rotation=90); ax.set_yticks(range(len(NUM)), NUM); ax.grid(False)
    ax.tick_params(labelsize=10.5)   # 20 rótulos por eje: es el mayor tamaño que cabe sin encimarse
    fig.colorbar(im, ax=ax, shrink=.82, pad=0.03, label="Spearman correlation (dimensionless)")
    guardar(fig, "fig_eda_spearman")

    fig, ax = plt.subplots(2, 1, figsize=(ANCHO, 7.6), sharex=True, gridspec_kw={"height_ratios": [20, 17]})
    for a, k, tag in zip(ax, ("vif_original", "vif_final"), ("(a) Original variables", "(b) Final model variables")):
        s = pd.Series(R[k]).sort_values()
        bb = a.barh(s.index, s.values, color=[GOLD if x > 10 else NAVY for x in s.values])
        for r, x in zip(bb, s.values):
            if x > 10:
                r.set_hatch("//"); r.set_edgecolor("white")
        a.axvline(10, ls="--", color="black"); a.grid(axis="y", visible=False); a.tick_params(axis="y", labelsize=10.5)
        a.text(0.97, 0.05, tag, transform=a.transAxes, ha="right", va="bottom", fontsize=12)
    ax[1].set_xlabel("Variance inflation factor (dimensionless)")
    guardar(fig, "fig_eda_vif")


# ------------------------------------------------------------------ modelo propuesto (metodología)
def corte_oblicuo():
    """Datos sintéticos con frontera diagonal: árbol por ejes frente a un corte de Fisher ponderado."""
    rng = np.random.default_rng(42); n0, n1 = 390, 110
    X = np.vstack([rng.multivariate_normal([-0.6, -0.6], [[1, 0.8], [0.8, 1]], n0),
                   rng.multivariate_normal([0.9, 0.9], [[1, 0.8], [0.8, 1]], n1)])
    y = np.r_[np.zeros(n0), np.ones(n1)]; w = np.where(y == 1, n0 / n1, 1.0)
    xx, yy = np.meshgrid(np.linspace(-3.5, 3.5, 400), np.linspace(-3.5, 3.5, 400)); grid = np.c_[xx.ravel(), yy.ravel()]
    za = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X, y, sample_weight=w).predict(grid).reshape(xx.shape)
    mu0, mu1 = X[y == 0].mean(0), X[y == 1].mean(0)
    D = X - np.where(y[:, None] == 1, mu1, mu0)
    v = np.linalg.solve((w[:, None] * D).T @ D / w.sum(), mu1 - mu0); v /= np.linalg.norm(v)
    z = X @ v; cands = np.sort(z)

    def gini(t):
        tot = 0.0
        for m in (z <= t, z > t):
            W = w[m].sum(); p = (w[m] * y[m]).sum() / W if W > 0 else 0
            tot += W * 2 * p * (1 - p)
        return tot
    t = cands[np.argmin([gini(c) for c in cands[1:-1]]) + 1]
    zb = (grid @ v > t).reshape(xx.shape)
    fig, axes = plt.subplots(1, 2, figsize=(ANCHO, 3.3), sharex=True, sharey=True)
    for ax, zz, tag in ((axes[0], za, "(a)"), (axes[1], zb, "(b)")):
        ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], colors=["#E8EEF6", "#F8E7C9"], alpha=0.9)
        if zz is zb:
            ax.contour(xx, yy, zz.astype(float), levels=[0.5], colors="#1F3A5F", linewidths=2)
        ax.scatter(*X[y == 0].T, s=9, marker="o", color="#1F3A5F", alpha=0.55, label="No default")
        ax.scatter(*X[y == 1].T, s=16, marker="^", color="#C77700", alpha=0.9, label="Default")
        ax.set_xlabel("PAY_0 (standardized)"); etiqueta(ax, tag, 0.03, 0.97)
    axes[0].set_ylabel("PAY_2 (standardized)"); axes[1].legend(frameon=False, loc="lower right", handletextpad=0.2)
    guardar(fig, "fig_csfof_split")


# ------------------------------------------------------------------ resultados
def f1_por_pliegue():
    T = pd.read_csv(AQUI / "metricas_pliegues.csv")
    orden = T.groupby("model").f1.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(ANCHO, 5.4))
    for i, m in enumerate(orden):
        v = T[T.model == m].f1.to_numpy(); col = GOLD if m == "cs_fof" else NAVY
        ax.boxplot(v, positions=[i], orientation="horizontal", widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor=col, alpha=0.25, edgecolor=col, hatch="//" if m == "cs_fof" else None),
                   medianprops=dict(color=col, lw=2), whiskerprops=dict(color=col), capprops=dict(color=col), flierprops=dict(markeredgecolor=col))
        ax.scatter(v, np.full(5, i), s=16, color=col, zorder=3, marker="D" if m == "cs_fof" else "o")
    ax.set_yticks(range(len(orden)), [EN[m] for m in orden]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("F1 of the default class (dimensionless)")
    guardar(fig, "fig_f1_folds")


def intervalos():
    R = pd.read_csv(AQUI / "pruebas_cs_fof.csv").sort_values("dif_f1", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(ANCHO, 5.0))
    ax.axvspan(-0.01, 0.01, color=GOLD, alpha=.18)
    for i, r in enumerate(R.itertuples()):
        sig = r.p_nb_holm < 0.05
        ax.plot([r.ic95_bajo, r.ic95_alto], [i, i], color=NAVY if sig else GREY, ls="-" if sig else "--")
        ax.scatter(r.dif_f1, i, color=NAVY if sig else GREY, zorder=3, s=40, marker="o" if sig else "s")
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(range(len(R)), [EN[m] for m in R.baseline]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("F1 difference: CS-FOF minus baseline")
    h = [Line2D([], [], color=NAVY, marker="o", label="Significant (Holm)"), Line2D([], [], color=GREY, ls="--", marker="s", label="Not significant"),
         matplotlib.patches.Patch(color=GOLD, alpha=.18, label="ROPE (±0.01)")]
    ax.legend(handles=h, frameon=False, loc="lower right")
    guardar(fig, "fig_diff_intervals")


def roc_pr_cal_cm():
    d = np.load(CF / "probabilidades_oof.npz"); y = d["y"]; mods = ("cs_fof", "catboost")
    fig, ax = plt.subplots(2, 2, figsize=(ANCHO, 6.0)); ax = ax.ravel()
    for m in mods:
        col, ls, mk = ESTILO[m]
        fpr, tpr, _ = roc_curve(y, d[m]); ax[0].plot(fpr, tpr, color=col, ls=ls)
        pr, rc, _ = precision_recall_curve(y, d[m]); ax[1].plot(rc, pr, color=col, ls=ls)
        fp, mp = calibration_curve(y, d[m], n_bins=10); ax[2].plot(mp, fp, color=col, ls=ls, marker=mk, ms=5)
        print(f"   {m}: AUC {roc_auc_score(y, d[m]):.3f}  AP {average_precision_score(y, d[m]):.3f}")
    ax[0].plot([0, 1], [0, 1], ls=":", color="k", lw=1.2); ax[0].set(xlabel="False positive rate", ylabel="True positive rate")
    ax[1].axhline(y.mean(), ls=":", color="k", lw=1.2); ax[1].set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))
    ax[2].plot([0, 1], [0, 1], ls=":", color="k", lw=1.2); ax[2].set(xlabel="Mean predicted probability", ylabel="Observed default rate")
    etiqueta(ax[0], "(a)"); etiqueta(ax[1], "(b)", 0.88); etiqueta(ax[2], "(c)")
    h = [Line2D([], [], color=ESTILO[m][0], ls=ESTILO[m][1], marker=ESTILO[m][2], ms=5, label=EN[m]) for m in mods]
    h.append(Line2D([], [], color="k", ls=":", lw=1.2, label="Reference"))
    ax[3].axis("off"); ax[3].legend(handles=h, frameon=False, loc="center", handlelength=3.2)
    guardar(fig, "fig_roc_pr_calibration")

    fig, ax = plt.subplots(1, 2, figsize=(ANCHO, 3.3))
    for a, m in zip(ax, mods):
        cm = confusion_matrix(y, d[m] >= 0.5)
        a.imshow(cm, cmap="Blues"); a.grid(False)
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13, color="white" if cm[i, j] > cm.max() / 2 else "black")
        a.set_xticks([0, 1], ["Pays", "Defaults"]); a.set_yticks([0, 1], ["Pays", "Defaults"], rotation=90, va="center")
        a.set_xlabel(f"Predicted by {EN[m]}")
    ax[0].set_ylabel("Actual class")
    guardar(fig, "fig_confusion")


def ablacion():
    A = pd.read_csv(AQUI / "ablacion_pliegues.csv")
    nombres = {"completo": "Full CS-FOF", "sin_oblicuo": "No oblique cut", "sin_costo": "No cost weighting",
               "todas_las_variables": "All node variables"}
    orden = ["completo", "sin_oblicuo", "todas_las_variables", "sin_costo"]
    fig, ax = plt.subplots(2, 1, figsize=(ANCHO, 4.6))
    for a, met, lab in ((ax[0], "f1", "F1 of the default class (dimensionless)"), (ax[1], "roc_auc", "ROC-AUC (dimensionless)")):
        g = A.groupby("variante")[met].agg(["mean", "std"]).reindex(orden)
        bb = a.barh(range(4), g["mean"], xerr=g["std"], color=[GOLD] + [NAVY] * 3, capsize=4, height=0.65)
        bb[0].set_hatch("//"); bb[0].set_edgecolor("white")
        a.set_yticks(range(4), [nombres[v] for v in orden]); a.invert_yaxis()
        a.set_xlim((g["mean"] - g["std"]).min() - 0.01, (g["mean"] + g["std"]).max() + 0.005)
        a.set_xlabel(lab); a.grid(axis="y", visible=False)
    guardar(fig, "fig_ablation")


def sensibilidad():
    ruido = pd.read_csv(RB / "ruido.csv"); tam = pd.read_csv(RB / "tamano.csv"); sem = pd.read_csv(RB / "semillas.csv")
    mods = ["cs_fof", "catboost", "xgboost", "lightgbm", "random_forest"]
    fig, ax = plt.subplots(2, 2, figsize=(ANCHO, 6.2)); ax = ax.ravel()
    for m in mods:
        col, ls, mk = ESTILO[m]
        g = tam[tam.model == m].groupby("nivel").f1.mean(); ax[0].plot(g.index * 100, g.values, color=col, ls=ls, marker=mk, ms=5)
        g = ruido[ruido.model == m].groupby("nivel").f1.mean(); ax[1].plot(g.index * 100, g.values, color=col, ls=ls, marker=mk, ms=5)
    ax[0].set(xlabel="Training data used (%)", ylabel="F1 (mean of 5 folds)"); ax[0].set_xscale("log")
    ax[0].set_xticks([10, 25, 50, 100], ["10", "25", "50", "100"]); ax[0].minorticks_off()
    ax[1].set(xlabel="Training labels flipped (%)", ylabel="F1 (mean of 5 folds)")
    ms = sem.groupby(["model", "semilla"]).f1.mean().reset_index()
    bp = ax[2].boxplot([ms[ms.model == m].f1 for m in mods], patch_artist=True, widths=0.6, medianprops=dict(color="k", lw=2))
    for caja, m in zip(bp["boxes"], mods):
        caja.set(facecolor=ESTILO[m][0], alpha=0.45, edgecolor=ESTILO[m][0])
    ax[2].set_xticks(range(1, 6), ["CS-FOF", "CatB.", "XGB.", "LGBM", "RF"]); ax[2].tick_params(axis="x", labelsize=11)
    ax[2].set(xlabel="Model (10 random seeds)", ylabel="F1 (mean of 5 folds)"); ax[2].grid(axis="x", visible=False)
    etiqueta(ax[0], "(a)", 0.84, 0.14); etiqueta(ax[1], "(b)", 0.84); etiqueta(ax[2], "(c)", 0.84)
    leyenda_modelos(ax[3], mods)
    guardar(fig, "fig_sensitivity")


def shap():
    W = pd.read_csv(RB / "shap_cs_fof_catboost.csv", index_col=0).head(10)[::-1]
    fig, ax = plt.subplots(figsize=(ANCHO, 4.6)); yy = np.arange(len(W))
    ax.barh(yy + .2, W.cs_fof, .4, color=GOLD, hatch="//", edgecolor="white", label="CS-FOF")
    ax.barh(yy - .2, W.catboost, .4, color=NAVY, label="CatBoost")
    ax.set_yticks(yy, W.index); ax.tick_params(axis="y", labelsize=10.5)
    ax.set_xlabel("Mean |SHAP value| (probability of default)"); ax.legend(frameon=False, loc="lower right"); ax.grid(axis="y", visible=False)
    guardar(fig, "fig_shap")


if __name__ == "__main__":
    for f in (eda, corte_oblicuo, f1_por_pliegue, intervalos, roc_pr_cal_cm, ablacion, sensibilidad, shap):
        f(); print("ok", f.__name__)
