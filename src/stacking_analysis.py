"""Pruebas estadísticas del estudio de stacking con validación repetida.

Lee ``<salida>/metricas.csv`` (de ``run_stacking_repeated.py``) y aplica los contrastes
pre-registrados en ``experiments/protocolo_stacking.md``:

- **H1 (principal):** ``stack_fwls_con`` frente a ``xgboost``.
- **H2 (principal):** ``stack_fwls_con`` frente a ``stack_fwls_sin`` (aporte de CS-FOF al ensamble).
  Holm sobre H1 y H2.
- **Exploratorios** (Holm aparte): los demás ensambles frente a XGBoost, CS-FOF frente a XGBoost y
  cada ensamble ``_con`` frente a su ``_sin``.

Pruebas por contraste, sobre las R x K diferencias pareadas de la métrica principal (F1 con el umbral
elegido en el entrenamiento):

- t corregida de Nadeau y Bengio (2003) para validación cruzada repetida: varianza multiplicada por
  ``1/(R K) + n_test/n_train``, ``R K - 1`` grados de libertad;
- t correlacionada bayesiana (Corani y Benavoli, 2015), misma posterior t, con ROPE de ±0.01;
- Wilcoxon de rangos con signo (sin corrección por dependencia, solo como referencia).

Uso::

    python -m src.stacking_analysis
    python -m src.stacking_analysis --salida results/stacking_prueba --metrica f1_05
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from scipy.stats import wilcoxon

from src.run_stacking_repeated import SALIDA_POR_DEFECTO

REFERENCIA = "xgboost"
PRINCIPALES = [("stack_fwls_con", "xgboost"), ("stack_fwls_con", "stack_fwls_sin")]
ROPE = 0.01


def nadeau_bengio(d: np.ndarray, n_train: float, n_test: float) -> dict:
    """t corregida para R x K diferencias pareadas (Nadeau y Bengio, 2003; Bouckaert y Frank, 2004)."""
    d = np.asarray(d, dtype=np.float64)
    J = len(d)
    var = d.var(ddof=1)
    se = np.sqrt((1 / J + n_test / n_train) * var)
    if se == 0:
        return {"t": np.inf * np.sign(d.mean()), "p": 0.0 if d.mean() else 1.0, "gl": J - 1, "ic_inf": d.mean(),
                "ic_sup": d.mean(), "se": 0.0}
    t = d.mean() / se
    crit = tdist.ppf(0.975, J - 1)
    return {"t": t, "p": 2 * tdist.sf(abs(t), J - 1), "gl": J - 1, "ic_inf": d.mean() - crit * se,
            "ic_sup": d.mean() + crit * se, "se": se}


def bayesiana_correlacionada(media: float, se: float, gl: int, rope: float = ROPE) -> tuple[float, float, float]:
    """P(peor), P(equivalente), P(mejor) con la posterior t de Corani y Benavoli (2015)."""
    if se == 0:
        return (float(media < -rope), float(abs(media) <= rope), float(media > rope))
    peor = tdist.cdf(-rope, gl, loc=media, scale=se)
    mejor = tdist.sf(rope, gl, loc=media, scale=se)
    return float(peor), float(1 - peor - mejor), float(mejor)


def holm(p: pd.Series) -> pd.Series:
    orden = p.sort_values()
    m = len(orden)
    ajust = np.maximum.accumulate([min(1.0, (m - i) * v) for i, v in enumerate(orden)])
    return pd.Series(ajust, index=orden.index).reindex(p.index)


def contrastar(tabla: pd.DataFrame, a: str, b: str, metrica: str) -> dict | None:
    piv = tabla.pivot_table(index=["repeticion", "pliegue"], columns="candidato", values=metrica)
    if a not in piv or b not in piv:
        return None
    d = (piv[a] - piv[b]).dropna().to_numpy()
    n_tr, n_te = tabla.n_train.mean(), tabla.n_test.mean()
    nb = nadeau_bengio(d, n_tr, n_te)
    peor, equiv, mejor = bayesiana_correlacionada(d.mean(), nb["se"], nb["gl"])
    try:
        p_w = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p_w = np.nan
    return {"a": a, "b": b, "n_pares": len(d), "dif_media": d.mean(), "ic95_inf": nb["ic_inf"],
            "ic95_sup": nb["ic_sup"], "gana_a": int((d > 0).sum()), "p_nadeau_bengio": nb["p"],
            "p_wilcoxon": p_w, "bayes_peor": peor, "bayes_rope": equiv, "bayes_mejor": mejor}


def analizar(salida: Path, metrica: str = "f1_umbral") -> tuple[pd.DataFrame, pd.DataFrame]:
    tabla = pd.read_csv(salida / "metricas.csv")
    cols = ["f1_umbral", "f1_05", "precision", "recall", "mcc", "roc_auc", "pr_auc", "segundos"]
    resumen = tabla.groupby("candidato")[cols].agg(["mean", "std"])
    resumen.columns = [f"{m}_{e}" for m, e in resumen.columns]
    resumen = resumen.sort_values(f"{metrica}_mean" if metrica in cols else "f1_umbral_mean", ascending=False)
    resumen.insert(0, "n_pliegues", tabla.groupby("candidato").size())

    candidatos = set(tabla.candidato)
    exploratorios = [(c, REFERENCIA) for c in sorted(candidatos) if c != REFERENCIA and (c, REFERENCIA) not in PRINCIPALES]
    exploratorios += [(c, c[:-4] + "_sin") for c in sorted(candidatos)
                      if c.endswith("_con") and (c, c[:-4] + "_sin") not in PRINCIPALES]
    filas = []
    for familia, pares in (("principal", PRINCIPALES), ("exploratorio", exploratorios)):
        bloque = [r | {"familia": familia} for a, b in pares if (r := contrastar(tabla, a, b, metrica))]
        if bloque:
            bloque = pd.DataFrame(bloque)
            bloque["p_holm"] = holm(bloque.p_nadeau_bengio)
            filas.append(bloque)
    contrastes = pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()
    if not contrastes.empty:
        contrastes["significativo"] = contrastes.p_holm < 0.05
        contrastes.insert(0, "metrica", metrica)
    return resumen, contrastes


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=SALIDA_POR_DEFECTO)
    ap.add_argument("--metrica", default="f1_umbral",
                    choices=["f1_umbral", "f1_05", "mcc", "roc_auc", "pr_auc", "precision", "recall"])
    args = ap.parse_args(argv)
    resumen, contrastes = analizar(args.salida, args.metrica)
    sufijo = "" if args.metrica == "f1_umbral" else f"_{args.metrica}"
    resumen.to_csv(args.salida / f"resumen{sufijo}.csv")
    contrastes.to_csv(args.salida / f"contrastes{sufijo}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"Resumen ({int(resumen.n_pliegues.max())} pliegues por candidato), métrica {args.metrica}:")
    print(resumen[[f"{args.metrica}_mean", f"{args.metrica}_std", "roc_auc_mean", "pr_auc_mean"]].round(4))
    print("\nContrastes (dif = a - b; p_holm dentro de cada familia):")
    print(contrastes[["familia", "a", "b", "dif_media", "ic95_inf", "ic95_sup", "gana_a", "p_nadeau_bengio",
                      "p_holm", "bayes_peor", "bayes_rope", "bayes_mejor", "significativo"]].round(4).to_string(index=False))
    print(f"\nGuardado en {args.salida}/resumen{sufijo}.csv y contrastes{sufijo}.csv")


if __name__ == "__main__":
    main()
