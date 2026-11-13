"""Cifras y figuras del apéndice ETL/EDA del paper, recalculadas con los 29 965 registros sin duplicados.

- Estadística descriptiva, comparación entre clases (Mann-Whitney U), chi-cuadrado y tasas por categoría.
- Correlación de Spearman (el único método de correlación del proyecto).
- VIF (datos winsorizados al 1-99 %, solo con fin diagnóstico) de las 20 variables originales y de las
  variables finales del modelo (las seis facturas resumidas en nivel medio, tendencia y meses con saldo a favor).

Uso (desde la raíz): python experiments/paper/apendice_eda.py
Salidas: experiments/paper/apendice_eda.json y figuras PDF en experiments/paper/figs/
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.outliers_influence import variance_inflation_factor  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
FIGS = AQUI / "figs"; FIGS.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "pdf.fonttype": 42, "font.size": 9})
GOLD, NAVY = "#B88F3D", "#36456D"
Y = "default_payment_next_month"
PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAMT = [f"PAY_AMT{i}" for i in range(1, 7)]
NUM = ["LIMIT_BAL", "AGE"] + PAY + BILL + PAMT


def cargar():
    d = pd.read_excel(RAIZ / "data" / "raw" / "default_of_credit_card_clients.xls", header=1)
    d = d.rename(columns={"default payment next month": Y}).drop(columns="ID")
    crudo = d.copy()
    d["EDUCATION"] = d["EDUCATION"].replace({0: 4, 5: 4, 6: 4}); d["MARRIAGE"] = d["MARRIAGE"].replace({0: 3})
    d = d.drop_duplicates().reset_index(drop=True)
    return crudo, d


def tukey(s):
    q1, q3 = s.quantile([.25, .75]); r = q3 - q1
    return float(((s < q1 - 1.5 * r) | (s > q3 + 1.5 * r)).mean())


def vif(df):
    X = df.copy()
    for c in X:
        lo, hi = X[c].quantile([.01, .99]); X[c] = X[c].clip(lo, hi)
    X = (X - X.mean()) / X.std(ddof=0)
    X.insert(0, "const", 1.0)
    return {c: float(variance_inflation_factor(X.values, i)) for i, c in enumerate(X.columns) if c != "const"}


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def main():
    crudo, d = cargar()
    R = {"n_crudo": len(crudo), "n_final": len(d), "duplicados": len(crudo) - len(crudo.drop_duplicates()),
         "educ_no_doc": int(crudo.EDUCATION.isin([0, 5, 6]).sum()), "marr_no_doc": int((crudo.MARRIAGE == 0).sum()),
         "nulos": int(d.isnull().sum().sum())}
    R["clase"] = {"0": int((d[Y] == 0).sum()), "1": int((d[Y] == 1).sum()), "tasa": float(d[Y].mean())}
    R["descriptiva"] = {c: dict(media=float(d[c].mean()), mediana=float(d[c].median()), de=float(d[c].std()),
                                asim=float(d[c].skew()), curt=float(d[c].kurt()), outliers=tukey(d[c])) for c in NUM}
    R["pay_codigos"] = {c: {str(k): float(v) for k, v in d[c].value_counts(normalize=True).loc[[-2, -1, 0]].items()} for c in PAY}
    R["bill_negativos_pct"] = {c: float((d[c] < 0).mean()) for c in BILL}
    R["filas_con_bill_negativo"] = int((d[BILL] < 0).any(axis=1).sum())
    g0, g1 = d[d[Y] == 0], d[d[Y] == 1]
    R["comparacion"] = {c: dict(media0=float(g0[c].mean()), media1=float(g1[c].mean()), mediana0=float(g0[c].median()),
                                mediana1=float(g1[c].median()), p_mw=float(stats.mannwhitneyu(g0[c], g1[c]).pvalue)) for c in NUM}
    R["chi2"] = {}
    for c in ("SEX", "EDUCATION", "MARRIAGE"):
        chi, p, dof, _ = stats.chi2_contingency(pd.crosstab(d[c], d[Y]))
        R["chi2"][c] = dict(chi2=float(chi), gl=int(dof), p=float(p),
                            tasas={str(k): float(v) for k, v in d.groupby(c)[Y].mean().items()},
                            pesos={str(k): float(v) for k, v in d[c].value_counts(normalize=True).items()})
    sp = d[NUM].corr("spearman")
    R["spearman_bill"] = [float(sp.loc[BILL, BILL].values[np.triu_indices(6, 1)].min()), float(sp.loc[BILL, BILL].values[np.triu_indices(6, 1)].max())]
    R["spearman_pay"] = [float(sp.loc[PAY, PAY].values[np.triu_indices(6, 1)].min()), float(sp.loc[PAY, PAY].values[np.triu_indices(6, 1)].max())]
    R["spearman_pamt"] = [float(sp.loc[PAMT, PAMT].values[np.triu_indices(6, 1)].min()), float(sp.loc[PAMT, PAMT].values[np.triu_indices(6, 1)].max())]
    R["vif_original"] = vif(d[NUM])
    bills = d[BILL].to_numpy(float); t = np.arange(5, -1, -1, float); tc = t - t.mean()
    fin = pd.DataFrame({"LIMIT_BAL_LOG": slog(d.LIMIT_BAL), "AGE": d.AGE, **{c: d[c] for c in PAY},
                        "BILL_AMT_MEAN_6M_LOG": slog(bills.mean(1)), "BILL_AMT_TREND_6M_LOG": slog(bills @ tc / (tc @ tc)),
                        "N_CREDIT_BALANCE_MONTHS": (bills < 0).sum(1), **{c + "_LOG": slog(d[c]) for c in PAMT}})
    R["vif_final"] = vif(fin)
    (AQUI / "apendice_eda.json").write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding="utf-8")

    # figuras
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    v = [R["clase"]["0"], R["clase"]["1"]]
    ax.bar(["No default (0)", "Default (1)"], v, color=[NAVY, GOLD])
    for i, x in enumerate(v):
        ax.text(i, x, f"{x:,}\n({x / sum(v):.1%})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Clients"); ax.set_ylim(0, max(v) * 1.2); ax.set_title("Target variable (n = 29,965)")
    fig.tight_layout(); fig.savefig(FIGS / "fig_eda_target.pdf"); plt.close(fig)

    # PAY_0 es discreta: barras agrupadas con la distribución dentro de cada clase (un diagrama de caja no la muestra bien)
    vals = [-2, -1, 0, 1, 2, 3, "4+"]; w = 0.4   # 4 a 8 meses se agrupan: son menos del 1 % de cada clase
    d0 = g0.PAY_0.clip(upper=4).value_counts(normalize=True).reindex(range(-2, 5), fill_value=0).to_numpy() * 100
    d1 = g1.PAY_0.clip(upper=4).value_counts(normalize=True).reindex(range(-2, 5), fill_value=0).to_numpy() * 100
    fig, ax = plt.subplots(figsize=(6.0, 3.1)); x = np.arange(len(vals))
    b0 = ax.bar(x - w / 2, d0, w, color=NAVY, label="No default"); b1 = ax.bar(x + w / 2, d1, w, color=GOLD, label="Default")
    for bars in (b0, b1):
        for r in bars:
            if r.get_height() >= 0.5:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.6, f"{r.get_height():.0f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, vals); ax.set_xlabel("PAY_0: September repayment status (months of delay; -2, -1, 0 = no delay)")
    ax.set_ylabel("Clients within each class (%)"); ax.set_ylim(0, max(d0.max(), d1.max()) * 1.15)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "fig_eda_pay0.pdf"); plt.close(fig)

    lab = {"1": "Graduate", "2": "University", "3": "High school", "4": "Others"}
    tasas = R["chi2"]["EDUCATION"]["tasas"]
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.bar([lab[k] for k in sorted(tasas)], [tasas[k] for k in sorted(tasas)], color=NAVY)
    ax.axhline(R["clase"]["tasa"], ls="--", color=GOLD, label=f"Overall rate ({R['clase']['tasa']:.1%})")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); ax.set_ylabel("Default rate"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "fig_eda_rate_educ.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(sp.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(NUM)), NUM, rotation=90, fontsize=7); ax.set_yticks(range(len(NUM)), NUM, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=.8, label="Spearman correlation"); ax.set_title("Spearman correlation among numeric predictors")
    fig.tight_layout(); fig.savefig(FIGS / "fig_eda_spearman.pdf"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.8), sharex=True)
    for a, (k, tit) in zip(ax, (("vif_original", "Original variables"), ("vif_final", "Final model variables"))):
        s = pd.Series(R[k]).sort_values()
        a.barh(s.index, s.values, color=[GOLD if x > 10 else NAVY for x in s.values]); a.axvline(10, ls="--", color="grey")
        a.set_title(tit); a.set_xlabel("VIF (winsorized 1-99%)"); a.tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "fig_eda_vif.pdf"); plt.close(fig)
    print(json.dumps({k: R[k] for k in ("n_crudo", "n_final", "duplicados", "clase", "spearman_bill", "spearman_pay", "filas_con_bill_negativo")}, indent=1))
    print("VIF final:", {k: round(v, 2) for k, v in sorted(R["vif_final"].items(), key=lambda x: -x[1])})


if __name__ == "__main__":
    main()
