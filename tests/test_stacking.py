"""Pruebas de los combinadores de stacking (src/stacking.py) y de las pruebas estadísticas
(src/stacking_analysis.py).

Uso, desde la raíz del proyecto:  python -m pytest tests/test_stacking.py -q
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.stacking import Promedio, SeleccionCaruana, StackingLogistico, StackingPorSegmento, logit, umbral_f1
from src.stacking_analysis import bayesiana_correlacionada, contrastar, holm, nadeau_bengio


@pytest.fixture
def datos():
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.random(n) < 0.22).astype(int)
    g = (rng.random(n) < 0.3).astype(float)
    bueno = np.clip(0.22 + 0.5 * (y - 0.22) + rng.normal(0, 0.15, n), 0, 1)
    ruido = rng.random(n)
    # "segmento": solo informativo cuando g = 1
    segmento = np.where(g == 1, np.clip(y * 0.8 + rng.normal(0.1, 0.1, n), 0, 1), rng.random(n))
    return np.column_stack([bueno, ruido, segmento]), y, g


def test_umbral_f1_es_optimo(datos):
    P, y, _ = datos
    s = P[:, 0]
    t = umbral_f1(y, s)
    mejor = f1_score(y, s >= t)
    for otro in np.quantile(s, np.linspace(0.05, 0.95, 37)):
        assert f1_score(y, s >= otro) <= mejor + 1e-12


def test_logit_acotado():
    assert np.all(np.isfinite(logit(np.array([0.0, 0.5, 1.0]))))


def test_caruana_prefiere_el_modelo_informativo(datos):
    P, y, _ = datos
    comb = SeleccionCaruana(n_iter=20).fit(P, y)
    assert comb.pesos_.sum() == pytest.approx(1.0)
    assert comb.pesos_[1] < comb.pesos_[0]  # nunca pesa más el ruido que el modelo bueno
    assert comb.predict_proba(P).shape == (len(y),)


def test_fwls_usa_el_segmento(datos):
    P, y, g = datos
    lr = StackingLogistico().fit(P, y)
    fw = StackingPorSegmento().fit(P, y, g)
    assert len(fw.pesos()) == 2 * P.shape[1] + 1
    # Con pesos por segmento el ajuste en la propia muestra no puede ser peor (modelo anidado).
    from sklearn.metrics import log_loss
    w = np.where(y == 1, (y == 0).sum() / (y == 1).sum(), 1.0)
    assert log_loss(y, fw.predict_proba(P, g), sample_weight=w) <= log_loss(y, lr.predict_proba(P), sample_weight=w) + 1e-6
    with pytest.raises(ValueError):
        fw.predict_proba(P)


def test_promedio(datos):
    P, y, _ = datos
    assert np.allclose(Promedio().fit(P, y).predict_proba(P), P.mean(axis=1))


def test_nadeau_bengio_es_mas_conservadora_que_la_t_ingenua():
    rng = np.random.default_rng(1)
    d = rng.normal(0.003, 0.005, 50)
    nb = nadeau_bengio(d, n_train=24000, n_test=6000)
    se_ingenuo = d.std(ddof=1) / np.sqrt(len(d))
    assert nb["se"] > se_ingenuo
    assert nb["ic_inf"] < d.mean() < nb["ic_sup"]
    assert nb["gl"] == 49


def test_bayesiana_suma_uno():
    p = bayesiana_correlacionada(0.004, 0.003, 49)
    assert sum(p) == pytest.approx(1.0)


def test_holm():
    p = pd.Series([0.01, 0.04, 0.03], index=list("abc"))
    assert holm(p).tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_contrastar_pares_por_repeticion_y_pliegue():
    filas = []
    for r in range(2):
        for k in range(5):
            for c, v in (("xgboost", 0.54), ("stack_fwls_con", 0.55)):
                filas.append(dict(repeticion=r, pliegue=k, candidato=c, f1_umbral=v + 0.001 * k,
                                  n_train=24000, n_test=6000))
    res = contrastar(pd.DataFrame(filas), "stack_fwls_con", "xgboost", "f1_umbral")
    assert res["n_pares"] == 10
    assert res["dif_media"] == pytest.approx(0.01)
    assert res["gana_a"] == 10


def test_reporte_de_progreso(tmp_path):
    from src.progreso import generar_html
    filas = [dict(repeticion=0, semilla=1000, pliegue=k, n_train=8, n_test=2, candidato=c, tipo="base",
                  f1_umbral=0.5 + 0.01 * k, roc_auc=0.78, segundos=30.0)
             for k in range(2) for c in ("xgboost", "tabm_fisher")]
    pd.DataFrame(filas).to_csv(tmp_path / "metricas.csv", index=False)
    (tmp_path / "configuracion.json").write_text('{"repeticiones": 2, "pliegues": 5}')
    pagina = generar_html(tmp_path, corriendo=True)
    assert "2 / 10" in pagina and "tabm_fisher" in pagina and "Corriendo" in pagina
    assert "p_holm" not in pagina  # sin significancia a mitad de camino
