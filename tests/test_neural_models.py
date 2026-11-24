"""Pruebas de los modelos neuronales (src/neural_models.py) y de la comparación final.

Las de TabM se saltan si no están instalados ``torch`` y ``tabm`` (la integración continua no los
instala). TabPFN no se prueba aquí porque sus pesos requieren una cuenta de Hugging Face.

Uso, desde la raíz del proyecto:  python -m pytest tests/test_neural_models.py -q
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from src.comparacion_final import comparar, recomendacion

PEQUENO = dict(k=4, n_blocks=1, d_block=32, max_epochs=8, patience=3, batch_size=64)


@pytest.fixture
def datos():
    X, y = make_classification(n_samples=600, n_features=8, n_informative=4, weights=[0.78], random_state=0)
    return X, y


def test_tabm_aprende_y_es_reproducible(datos):
    pytest.importorskip("torch")
    pytest.importorskip("tabm")
    from src.neural_models import TabMClassifier
    X, y = datos
    a = TabMClassifier(class_weight="balanced", random_state=1, device="cpu", **PEQUENO).fit(X, y)
    p = a.predict_proba(X)
    assert p.shape == (len(y), 2) and np.allclose(p.sum(1), 1)
    assert roc_auc_score(y, p[:, 1]) > 0.75
    b = clone(a).fit(X, y)
    assert np.allclose(p, b.predict_proba(X), atol=1e-6)


def test_tabm_fisher_agrega_las_distancias(datos):
    pytest.importorskip("torch")
    pytest.importorskip("tabm")
    from src.neural_models import TabMClassifier, cortes_fisher
    X, y = datos
    m = TabMClassifier(class_weight="balanced", fisher_features=6, fisher_trees=10, random_state=0, device="cpu",
                       **PEQUENO).fit(X, y)
    assert 0 < len(m.cortes_) <= 6
    assert all(len(c) > 1 for c, _, _ in m.cortes_)          # solo cortes oblicuos
    assert m._entrada(X).shape[1] == X.shape[1] + len(m.cortes_)
    assert len(cortes_fisher(m.bosque_, max_prof=0)) <= m.fisher_trees  # a lo sumo uno por raíz
    with pytest.raises(ValueError):
        m.predict_proba(X[:, :3])


def _tabla(dif_stack, dif_red):
    filas = []
    for r in range(4):
        for k in range(5):
            ruido = 0.0005 * (((r * 5 + k) % 3) - 1)  # media cero
            for c, v in (("xgboost", 0.54), ("stack_fwls_con", 0.54 + dif_stack + ruido)):
                filas.append(dict(fuente="s", repeticion=r, pliegue=k, candidato=c, f1_umbral=v, n_train=24000, n_test=6000))
            for c, v in (("xgboost", 0.54), ("tabm_fisher", 0.54 + dif_red + ruido), ("tabm", 0.54)):
                filas.append(dict(fuente="r", repeticion=r, pliegue=k, candidato=c, f1_umbral=v, n_train=24000, n_test=6000))
    t = pd.DataFrame(filas)
    t.loc[(t.fuente == "s") & (t.candidato == "xgboost"), "candidato"] = "xgboost_stacking"
    t.loc[(t.fuente == "r") & (t.candidato == "xgboost"), "candidato"] = "xgboost_redes"
    return t


def test_regla_de_decision():
    assert "STACKING" in recomendacion(comparar(_tabla(0.02, 0.0)))
    assert "TabM-Fisher" in recomendacion(comparar(_tabla(0.0, 0.02)))
    assert "Ninguna" in recomendacion(comparar(_tabla(0.0, 0.0)))
