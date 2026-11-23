"""Combinadores de nivel 1 para el estudio de stacking (``run_stacking_repeated.py``).

Todos reciben las probabilidades *out-of-fold* (OOF) de los modelos base, calculadas solo con el
entrenamiento del pliegue externo, y nunca ven el pliegue de prueba.

- ``StackingLogistico``: regresión logística sobre los logits de los modelos base.
- ``StackingPorSegmento``: *feature-weighted linear stacking* (Sill et al., 2009) con una sola
  meta-variable, ``g = 1{PAY_0 > 0}`` (cliente con atraso en septiembre). El meta-modelo usa
  ``[L, L * g, g]``, así que aprende pesos distintos para los clientes al día y para los atrasados.
  Responde al análisis de errores del artículo: XGBoost acierta más entre los clientes al día y
  CS-FOF empata o gana entre los atrasados.
- ``SeleccionCaruana``: selección de ensambles con reemplazo (Caruana et al., 2004), la que usan
  AutoGluon y TabArena. Promedio ponderado de probabilidades; pesos = frecuencia de selección.
- ``Promedio``: promedio simple de probabilidades (referencia sin parámetros).

``umbral_f1`` elige el umbral que maximiza el F1 sobre puntajes OOF, igual para todos los candidatos.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def umbral_f1(y: np.ndarray, s: np.ndarray) -> float:
    """Umbral ``t`` que maximiza el F1 de la regla ``s >= t`` sobre (y, s)."""
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = np.where(prec[:-1] + rec[:-1] > 0, 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12), 0.0)
    return float(thr[int(np.argmax(f1))])


class StackingLogistico:
    """Regresión logística (pesos de clase balanceados) sobre los logits de los modelos base."""

    def __init__(self, C: float = 1.0):
        self.C = C

    def _diseno(self, P: np.ndarray, g: np.ndarray | None) -> np.ndarray:
        return logit(P)

    def fit(self, P: np.ndarray, y: np.ndarray, g: np.ndarray | None = None):
        self.lr_ = LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000)
        self.lr_.fit(self._diseno(P, g), y)
        return self

    def predict_proba(self, P: np.ndarray, g: np.ndarray | None = None) -> np.ndarray:
        return self.lr_.predict_proba(self._diseno(P, g))[:, 1]

    def pesos(self) -> list[float]:
        return [float(c) for c in self.lr_.coef_.ravel()]


class StackingPorSegmento(StackingLogistico):
    """FWLS con la meta-variable ``g``: pesos distintos para clientes al día (g=0) y atrasados (g=1)."""

    def _diseno(self, P: np.ndarray, g: np.ndarray | None) -> np.ndarray:
        if g is None:
            raise ValueError("StackingPorSegmento necesita la meta-variable g (1{PAY_0 > 0}).")
        L = logit(P)
        g = np.asarray(g, dtype=np.float64).reshape(-1, 1)
        return np.hstack([L, L * g, g])


class SeleccionCaruana:
    """Selección de ensambles con reemplazo (Caruana et al., 2004).

    Empieza por el mejor modelo y en cada paso agrega (con reemplazo) el modelo que más mejora la
    métrica del promedio. Por defecto optimiza la precisión media (AUC-PR), que no depende del umbral
    y se centra en la clase minoritaria; el umbral se elige después con ``umbral_f1``.
    """

    def __init__(self, n_iter: int = 50, metrica=average_precision_score):
        self.n_iter = n_iter
        self.metrica = metrica

    def fit(self, P: np.ndarray, y: np.ndarray, g: np.ndarray | None = None):
        P = np.asarray(P, dtype=np.float64)
        m = P.shape[1]
        cuentas = np.zeros(m)
        j = int(np.argmax([self.metrica(y, P[:, k]) for k in range(m)]))
        cuentas[j] += 1
        actual, n = P[:, j].copy(), 1
        for _ in range(self.n_iter - 1):
            puntajes = [self.metrica(y, (actual * n + P[:, k]) / (n + 1)) for k in range(m)]
            j = int(np.argmax(puntajes))
            actual = (actual * n + P[:, j]) / (n + 1)
            n += 1
            cuentas[j] += 1
        self.pesos_ = cuentas / cuentas.sum()
        return self

    def predict_proba(self, P: np.ndarray, g: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(P, dtype=np.float64) @ self.pesos_

    def pesos(self) -> list[float]:
        return [float(w) for w in self.pesos_]


class Promedio:
    """Promedio simple de probabilidades."""

    def fit(self, P, y, g=None):
        self.m_ = np.asarray(P).shape[1]
        return self

    def predict_proba(self, P, g=None):
        return np.asarray(P, dtype=np.float64).mean(axis=1)

    def pesos(self):
        return [1.0 / self.m_] * self.m_
