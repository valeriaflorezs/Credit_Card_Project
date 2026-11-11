"""El algoritmo genético del proyecto, con sus mismos operadores, resuelve las funciones de De Jong."""
import numpy as np
import pytest

pytest.importorskip("deap")
pytest.importorskip("optuna")
from src.dejong import FUNCIONES, aleatoria, f1_esfera, f2_rosenbrock, f3_escalon, f4_cuartica, f5_shekel, genetico


def test_minimos_conocidos():
    assert f1_esfera(np.zeros(3)) == 0
    assert f2_rosenbrock(np.ones(2)) == 0
    assert f3_escalon(np.full(5, -5.12)) == -30
    assert f4_cuartica(np.zeros(30)) == 0
    assert abs(f5_shekel(np.array([-32.0, -32.0])) - 0.998004) < 1e-4


def test_converge_en_la_esfera_con_presupuesto_amplio():
    hist, _ = genetico("F1 esfera", semilla=0, population_size=50, n_generations=50)
    assert hist[-1] < 0.05
    assert np.all(np.diff(hist) <= 0)   # el mejor hasta ahora nunca empeora (elitismo)


def test_supera_a_la_aleatoria_en_la_esfera_con_el_presupuesto_del_proyecto():
    gana = 0
    for s in range(5):
        g, n = genetico("F1 esfera", s)
        gana += g[-1] < aleatoria("F1 esfera", s, n)[-1]
    assert gana >= 4


def test_reproducible_con_la_misma_semilla():
    a, _ = genetico("F2 Rosenbrock", semilla=3)
    b, _ = genetico("F2 Rosenbrock", semilla=3)
    np.testing.assert_array_equal(a, b)
