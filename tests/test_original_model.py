"""Pruebas unitarias de CS-FOF (src/original_model.py).

Cubren las exigencias del Entregable 3: (i) aprende en datos con solución
conocida, (ii) reproducibilidad con la misma semilla, (iii) entradas inválidas,
(iv) compatibilidad con sklearn.base.clone y (v) verificación numérica de las
propiedades teóricas de la metodología.

Uso, desde la raíz del proyecto:  python -m pytest tests/test_original_model.py -q
"""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.original_model import (
    CSFisherObliqueForest,
    _gini_ponderado,
    _pesos_de_clase,
    direccion_fisher_por_nodo,
)


@pytest.fixture
def datos():
    return make_classification(n_samples=800, n_features=10, n_informative=5,
                               weights=[0.78], random_state=0)


# (i) aprende ------------------------------------------------------------------
def test_aprende_mejor_que_azar(datos):
    X, y = datos
    m = CSFisherObliqueForest(n_estimators=20, random_state=0).fit(X, y)
    assert (m.predict(X) == y).mean() > 0.85


def test_frontera_diagonal_la_capta_un_solo_corte_oblicuo():
    """Con una frontera x1 + x2 = 0, un árbol de profundidad 1 con corte oblicuo
    acierta mucho más que uno que solo puede cortar por ejes."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    kw = dict(n_estimators=1, max_depth=1, min_samples_leaf=5, max_features=None, bootstrap=False, random_state=0)
    acc_obl = (CSFisherObliqueForest(**kw).fit(X, y).predict(X) == y).mean()
    acc_eje = (CSFisherObliqueForest(**kw, oblique=False).fit(X, y).predict(X) == y).mean()
    assert acc_obl > 0.97 and acc_obl > acc_eje + 0.15


# (ii) reproducibilidad --------------------------------------------------------
def test_reproducible_con_misma_semilla(datos):
    X, y = datos
    p1 = CSFisherObliqueForest(n_estimators=8, random_state=1).fit(X, y).predict_proba(X)
    p2 = CSFisherObliqueForest(n_estimators=8, random_state=1).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(p1, p2)


def test_n_jobs_no_cambia_el_resultado(datos):
    X, y = datos
    p1 = CSFisherObliqueForest(n_estimators=8, random_state=3, n_jobs=1).fit(X, y).predict_proba(X)
    p2 = CSFisherObliqueForest(n_estimators=8, random_state=3, n_jobs=2).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(p1, p2)


# (iii) entradas inválidas -----------------------------------------------------
def test_rechaza_nan(datos):
    X, y = datos
    X = X.copy(); X[0, 0] = np.nan
    with pytest.raises(ValueError):
        CSFisherObliqueForest().fit(X, y)


def test_rechaza_mas_de_dos_clases():
    X, y = make_classification(n_samples=300, n_classes=3, n_informative=4, random_state=0)
    with pytest.raises(ValueError):
        CSFisherObliqueForest().fit(X, y)


def test_rechaza_numero_de_columnas_distinto(datos):
    X, y = datos
    m = CSFisherObliqueForest(n_estimators=3, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        m.predict(X[:, :5])


# (iv) compatibilidad con scikit-learn -----------------------------------------
def test_clone_conserva_hiperparametros():
    m = CSFisherObliqueForest(n_estimators=7, eps=0.3, oblique_features=0.5, n_oblique=2)
    assert clone(m).get_params() == m.get_params()


def test_funciona_en_pipeline_y_cross_val(datos):
    X, y = datos
    pipe = make_pipeline(StandardScaler(), CSFisherObliqueForest(n_estimators=5, random_state=0))
    scores = cross_val_score(pipe, X, y, cv=3, scoring="f1")
    assert np.all(np.isfinite(scores))


def test_probabilidades_validas(datos):
    X, y = datos
    P = CSFisherObliqueForest(n_estimators=5, random_state=0).fit(X, y).predict_proba(X)
    assert P.shape == (len(X), 2)
    np.testing.assert_allclose(P.sum(1), 1.0)
    assert (P >= 0).all() and (P <= 1).all()


# (v) propiedades teóricas -----------------------------------------------------
def _gini_de_la_raiz(modelo, X, y, w):
    """Gini ponderado G(L, R) de la partición que hace la raíz del único árbol."""
    a = modelo.arboles_[0]
    Z = (X - modelo.media_) / modelo.escala_
    f, c = a.feat_[0], a.coef_[0]
    proy = (Z[:, f[f >= 0]] * c[f >= 0]).sum(1)
    L = proy <= a.thr_[0]
    return _gini_ponderado(w[L].sum(), (w * y)[L].sum(), w[~L].sum(), (w * y)[~L].sum())


def test_prop1_corte_oblicuo_nunca_peor_que_ejes(datos):
    """Proposición 1: con las mismas variables sorteadas, la impureza del corte
    elegido por CS-FOF no supera la del mejor corte por ejes."""
    X, y = datos
    w = _pesos_de_clase(y, "balanced")
    kw = dict(n_estimators=1, max_depth=1, min_samples_leaf=10, bootstrap=False,
              class_weight="balanced", random_state=5)
    g_csfof = _gini_de_la_raiz(CSFisherObliqueForest(**kw).fit(X, y), X, y, w)
    g_ejes = _gini_de_la_raiz(CSFisherObliqueForest(**kw, oblique=False).fit(X, y), X, y, w)
    assert g_csfof <= g_ejes + 1e-9


def test_prop2_fisher_maximiza_el_criterio(datos):
    """Proposición 2: v maximiza J(u) = (u^T (mu1 - mu0))^2 / (u^T S_w u)."""
    X, y = datos
    y = y.astype(float)
    w = _pesos_de_clase(y, "balanced")
    nodo = np.zeros(len(y), dtype=np.int64)
    V, valido = direccion_fisher_por_nodo(X, y, w, nodo, 1, eps=0.0)
    assert valido[0]
    v = V[0]
    mu1 = (w * y) @ X / (w * y).sum()
    mu0 = (w * (1 - y)) @ X / (w * (1 - y)).sum()
    D = X - np.where(y[:, None] == 1, mu1, mu0)
    Sw = (w[:, None] * D).T @ D / w.sum()
    d = mu1 - mu0
    J = lambda u: (u @ d) ** 2 / (u @ Sw @ u)
    rng = np.random.default_rng(0)
    assert all(J(v) >= J(rng.normal(size=len(d))) - 1e-9 for _ in range(300))


def test_prop3_termina_y_respeta_hojas_minimas(datos):
    """Proposición 3: el árbol termina y tiene a lo sumo floor(n / h) hojas."""
    X, y = datos
    h = 25
    m = CSFisherObliqueForest(n_estimators=1, max_depth=50, min_samples_leaf=h,
                              bootstrap=False, random_state=0).fit(X, y)
    hojas = (m.arboles_[0].izq_ < 0).sum()
    assert hojas <= len(y) // h


def test_class_weight_cambia_las_probabilidades(datos):
    X, y = datos
    p_sin = CSFisherObliqueForest(n_estimators=5, random_state=0).fit(X, y).predict_proba(X)[:, 1]
    p_con = CSFisherObliqueForest(n_estimators=5, random_state=0, class_weight="balanced").fit(X, y).predict_proba(X)[:, 1]
    assert p_con.mean() > p_sin.mean()   # más peso a los morosos -> mayor probabilidad media de incumplir
