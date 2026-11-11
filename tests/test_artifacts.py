"""Prueba de ida y vuelta de los artefactos (Entregable 3, sección 4.2).

``src/train_artifacts.py`` guardó 200 clientes fijos y las probabilidades que daba cada modelo en memoria
justo antes de serializarlo. Aquí se carga cada ``.joblib`` y se exige que prediga exactamente lo mismo.
Si falta la librería de un modelo (por ejemplo, catboost), su prueba se omite en vez de fallar.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ART = Path(__file__).resolve().parents[1] / "artifacts"
INDICE = json.loads((ART / "indice.json").read_text(encoding="utf-8"))
LIBRERIA = {"catboost": "catboost", "xgboost": "xgboost", "lightgbm": "lightgbm"}


@pytest.fixture(scope="module")
def muestra():
    d = np.load(ART / "ida_y_vuelta.npz", allow_pickle=False)
    X = pd.DataFrame(d["X"], columns=[str(c) for c in d["columnas"]])
    return X, d


def test_indice_tiene_los_10_modelos():
    assert len(INDICE) == 10
    assert all(m["prueba_ida_y_vuelta"] for m in INDICE)


@pytest.mark.parametrize("entrada", INDICE, ids=[m["modelo"] for m in INDICE])
def test_ida_y_vuelta(entrada, muestra):
    pytest.importorskip("imblearn")
    if entrada["modelo"] in LIBRERIA:
        pytest.importorskip(LIBRERIA[entrada["modelo"]])
    X, d = muestra
    modelo = joblib.load(ART / entrada["archivo"])
    p = modelo.predict_proba(X)[:, 1]
    np.testing.assert_allclose(p, d[f"p_{entrada['modelo']}"], rtol=0, atol=1e-9)  # misma máquina: idénticas; otra máquina (CI): tolerancia numérica


@pytest.mark.parametrize("entrada", INDICE, ids=[m["modelo"] for m in INDICE])
def test_metadata_completa(entrada):
    m = json.loads((ART / entrada["metadata"]).read_text(encoding="utf-8"))
    for clave in ("semilla", "hiperparametros_finales", "columnas_entrada", "metricas_bucle_externo_5x5", "versiones"):
        assert clave in m, f"falta {clave} en {entrada['metadata']}"
