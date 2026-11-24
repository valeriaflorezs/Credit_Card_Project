"""Comparación final: stacking (sin redes) frente a redes neuronales, con las mismas particiones.

Lee las dos corridas de ``run_stacking_repeated.py`` (mismas semillas 1000-1009, mismos pliegues):

- ``results/stacking_repeated/metricas.csv`` (compañera: stacking con CS-FOF);
- ``results/redes_repeated/metricas.csv`` (redes: TabM-Fisher, TabM, TabPFN y XGBoost).

Aplica los contrastes de ``experiments/protocolo_redes.md``:

- **Familia final** (Holm sobre las dos): ``stack_fwls_con - xgboost`` y ``tabm_fisher - xgboost``. Cada
  opción se compara con el XGBoost de **su propia corrida**, así una diferencia de versiones de librerías
  entre computadores (por ejemplo, Colab frente al portátil) no sesga el contraste.
- **Secundarios** (Holm aparte): ``tabm_fisher - tabm`` (aporte de las variables de Fisher),
  ``tabpfn - xgboost``, ``tabm - xgboost`` y la comparación directa ``stack_fwls_con - tabm_fisher``.

Imprime la recomendación según la regla pre-registrada (sección 7 del protocolo).

Uso::

    python -m src.comparacion_final
    python -m src.comparacion_final --stacking results/stacking_repeated --redes results/redes_repeated
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import config
from src.stacking_analysis import contrastar, holm

STACK, RED = "stack_fwls_con", "tabm_fisher"


def cargar(stacking: Path, redes: Path) -> pd.DataFrame:
    a = pd.read_csv(stacking / "metricas.csv")
    b = pd.read_csv(redes / "metricas.csv")
    a["candidato"] = a.candidato.replace({"xgboost": "xgboost_stacking"})
    b["candidato"] = b.candidato.replace({"xgboost": "xgboost_redes"})
    comunes = set(zip(a.repeticion, a.pliegue)) & set(zip(b.repeticion, b.pliegue))
    tabla = pd.concat([a, b], ignore_index=True)
    tabla = tabla[[par in comunes for par in zip(tabla.repeticion, tabla.pliegue)]]
    # Las particiones deben ser idénticas: mismo tamaño de prueba en cada (repetición, pliegue).
    tam = tabla.groupby(["repeticion", "pliegue"]).n_test.nunique()
    if (tam > 1).any():
        raise ValueError("Las dos corridas no usan las mismas particiones (n_test distinto). Revisen semillas y datos.")
    return tabla


def comparar(tabla: pd.DataFrame, metrica: str = "f1_umbral") -> pd.DataFrame:
    familias = {
        "final": [(STACK, "xgboost_stacking"), (RED, "xgboost_redes")],
        "secundario": [(RED, "tabm"), ("tabpfn", "xgboost_redes"), ("tabm", "xgboost_redes"), (STACK, RED)],
    }
    bloques = []
    for familia, pares in familias.items():
        filas = [r | {"familia": familia} for a, b in pares if (r := contrastar(tabla, a, b, metrica))]
        if filas:
            df = pd.DataFrame(filas)
            df["p_holm"] = holm(df.p_nadeau_bengio)
            df["significativo"] = (df.p_holm < 0.05) & (df.dif_media > 0)
            bloques.append(df)
    return pd.concat(bloques, ignore_index=True)


def recomendacion(c: pd.DataFrame) -> str:
    def sig(a, b):
        f = c[(c.a == a) & (c.b == b)]
        return bool(len(f) and f.significativo.iloc[0])

    gana_stack = sig(STACK, "xgboost_stacking")
    gana_red = sig(RED, "xgboost_redes")
    if gana_stack and not gana_red:
        return "Quedarse con el STACKING con CS-FOF: supera a XGBoost y no tiene el tope de 4.5."
    if gana_red and not gana_stack:
        return "Quedarse con TabM-Fisher: es la única opción que supera a XGBoost (nota máxima 4.5)."
    if gana_red and gana_stack:
        if sig(RED, STACK):  # la red es significativamente mejor que el stacking
            return ("Las dos superan a XGBoost y TabM-Fisher también supera al stacking: decidir si la mejora "
                    "compensa el tope de 4.5.")
        return "Las dos superan a XGBoost sin diferencia entre ellas: quedarse con el STACKING (sin tope de nota)."
    return ("Ninguna supera a XGBoost con significancia: reportar el empate y quedarse con CS-FOF / stacking "
            "(sin tope de nota).")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacking", type=Path, default=config.RESULTS_DIR / "stacking_repeated")
    ap.add_argument("--redes", type=Path, default=config.RESULTS_DIR / "redes_repeated")
    ap.add_argument("--metrica", default="f1_umbral")
    ap.add_argument("--salida", type=Path, default=config.RESULTS_DIR / "comparacion_final.csv")
    args = ap.parse_args(argv)

    tabla = cargar(args.stacking, args.redes)
    resumen = (tabla.groupby("candidato")[[args.metrica, "roc_auc", "pr_auc"]].agg(["mean", "std"])
               .sort_values((args.metrica, "mean"), ascending=False))
    c = comparar(tabla, args.metrica)
    c.to_csv(args.salida, index=False)
    pd.set_option("display.width", 200)
    print(f"{tabla.groupby(['repeticion', 'pliegue']).ngroups} pliegues comunes. Métrica: {args.metrica}\n")
    print(resumen.round(4))
    print("\nContrastes (dif = a - b):")
    print(c[["familia", "a", "b", "dif_media", "ic95_inf", "ic95_sup", "p_nadeau_bengio", "p_holm",
             "bayes_peor", "bayes_rope", "bayes_mejor", "significativo"]].round(4).to_string(index=False))
    print(f"\nRecomendación (regla pre-registrada): {recomendacion(c)}")
    print(f"Guardado en {args.salida}")


if __name__ == "__main__":
    main()
