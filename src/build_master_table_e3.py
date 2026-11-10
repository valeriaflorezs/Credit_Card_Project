"""Tabla maestra del Entregable 3: las 112 combinaciones 5 x 5 de los 7 modelos base,
las de CS-FOF y, si ya existen, las de LightGBM y CatBoost.

Todas usan los mismos 5 pliegues externos (semilla 42) y desviaciones con ddof=1.
Salida: results/master_table_e3.csv

Uso (desde la raíz)::

    python -m src.build_master_table_e3
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config

BASE = config.RESULTS_DIR / "master_table_112.csv"
NUEVOS = {
    "cs_fof": config.RESULTS_DIR / "experiments_5x5__cs_fof.csv",
    "boosting_recientes": config.RESULTS_DIR / "experiments_5x5__boosting_recientes.csv",
}
SALIDA = config.RESULTS_DIR / "master_table_e3.csv"
METRICAS = ["f1", "precision", "recall", "roc_auc", "accuracy"]


def _normalizar(df: pd.DataFrame, fuente: str) -> pd.DataFrame:
    """Recalcula media y desviación (ddof=1) desde los valores por pliegue."""
    df = df.copy()
    for m in METRICAS:
        col = f"{m}_per_fold"
        if col in df:
            v = df[col].apply(json.loads)
            df[f"{m}_mean"] = v.apply(np.mean)
            df[f"{m}_std"] = v.apply(lambda x: float(np.std(x, ddof=1)))
    df["source"] = fuente
    return df


if __name__ == "__main__":
    partes = [pd.read_csv(BASE)]
    for nombre, ruta in NUEVOS.items():
        if ruta.exists():
            partes.append(_normalizar(pd.read_csv(ruta), f"local_cpu_{nombre}"))
            print(f"agregado: {ruta.name}")
        else:
            print(f"todavía no existe: {ruta.name}")
    tabla = pd.concat(partes, ignore_index=True, sort=False)
    tabla = tabla.sort_values("f1_mean", ascending=False).reset_index(drop=True)
    tabla.to_csv(SALIDA, index=False)
    mejores = tabla.loc[tabla.groupby("model").f1_mean.idxmax()].sort_values("f1_mean", ascending=False)
    print(f"\n{len(tabla)} combinaciones en {SALIDA.name}. Mejor combinación por modelo:")
    print(mejores[["model", "technique", "method", "f1_mean", "f1_std", "roc_auc_mean"]].round(4).to_string(index=False))
