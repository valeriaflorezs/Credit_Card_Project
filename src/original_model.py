"""CS-FOF: Cost-Sensitive Fisher Oblique Forest (modelo original del proyecto).

Bosque aleatorio cuyos nodos pueden cortar por una combinación lineal de
variables. En cada nodo compiten:

- los cortes por ejes sobre ``q`` variables sorteadas (como en Random Forest), y
- un corte oblicuo por la dirección discriminante de Fisher, ponderada por el
  costo del incumplimiento y calculada SOLO con los clientes de ese nodo.

El nodo se queda con el corte de menor impureza de Gini ponderada.

Implementación propia: solo usa ``numpy`` como herramienta de cálculo (la
clase base de ``scikit-learn`` solo aporta la interfaz ``fit/predict``).

Decisiones de implementación (no cambian el modelo, solo su costo):

1. El árbol crece **por niveles**: todos los nodos de una profundidad se
   procesan a la vez con operaciones vectorizadas, en vez de un nodo por
   llamada recursiva.
2. Los umbrales se buscan con **histogramas** de ``n_bins`` cuantiles, como en
   LightGBM o el modo ``hist`` de XGBoost: los cortes por ejes usan los
   cuantiles de cada variable (calculados una vez por ajuste) y el corte
   oblicuo usa los cuantiles de la proyección dentro de cada nodo.
3. El *bootstrap* se representa con pesos (cuántas veces sale cada cliente)
   en vez de duplicar filas.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _n_features_por_nodo(max_features, p: int) -> int:
    """Número q de variables sorteadas en cada nodo."""
    if max_features is None:
        return p
    if max_features == "sqrt":
        return max(1, int(np.sqrt(p)))
    if max_features == "log2":
        return max(1, int(np.log2(p)))
    if isinstance(max_features, (float, np.floating)) and 0 < max_features <= 1:
        return max(1, int(round(max_features * p)))
    if isinstance(max_features, (int, np.integer)) and max_features >= 1:
        return min(int(max_features), p)
    raise ValueError(f"max_features no válido: {max_features!r}")


def _pesos_de_clase(y: np.ndarray, class_weight) -> np.ndarray:
    """Peso w_i de cada cliente (ecuación de pesos de la metodología).

    ``class_weight=None``: todos pesan 1 (c = 1).
    ``class_weight="balanced"``: el moroso pesa c = n0 / n1.
    ``class_weight={0: a, 1: b}``: pesos explícitos.
    """
    if class_weight is None:
        return np.ones(len(y))
    if class_weight == "balanced":
        n1 = max(int(y.sum()), 1)
        n0 = len(y) - n1
        return np.where(y == 1, n0 / n1, 1.0)
    if isinstance(class_weight, dict):
        return np.where(y == 1, float(class_weight.get(1, 1.0)), float(class_weight.get(0, 1.0)))
    raise ValueError(f"class_weight no válido: {class_weight!r}")


def _gini_ponderado(WL, W1L, WR, W1R):
    """G(L, R) = W_L 2 p_L (1 - p_L) + W_R 2 p_R (1 - p_R), sin dividir por cero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pL = np.where(WL > 0, W1L / WL, 0.0)
        pR = np.where(WR > 0, W1R / WR, 0.0)
    return WL * 2 * pL * (1 - pL) + WR * 2 * pR * (1 - pR)


def _mejor_umbral_por_histograma(nodo, bins, w, wy, cnt, n_nodos, n_bins, min_leaf):
    """Mejor corte «bin <= b» de cada nodo a partir de histogramas ponderados.

    Devuelve, por nodo, (impureza mínima, bin óptimo). Impureza = +inf si
    ningún corte deja al menos ``min_leaf`` observaciones a cada lado.
    """
    clave = nodo * n_bins + bins
    tam = n_nodos * n_bins
    H_w = np.bincount(clave, weights=w, minlength=tam).reshape(n_nodos, n_bins)
    H_wy = np.bincount(clave, weights=wy, minlength=tam).reshape(n_nodos, n_bins)
    H_c = np.bincount(clave, weights=cnt, minlength=tam).reshape(n_nodos, n_bins)
    WL = np.cumsum(H_w, axis=1)[:, :-1]
    W1L = np.cumsum(H_wy, axis=1)[:, :-1]
    CL = np.cumsum(H_c, axis=1)[:, :-1]
    W, W1, C = H_w.sum(1, keepdims=True), H_wy.sum(1, keepdims=True), H_c.sum(1, keepdims=True)
    G = _gini_ponderado(WL, W1L, W - WL, W1 - W1L)
    G[(CL < min_leaf) | (C - CL < min_leaf)] = np.inf
    b = np.argmin(G, axis=1)
    return G[np.arange(n_nodos), b], b


def direccion_fisher_por_nodo(Xs, y, w, nodo, n_nodos, eps):
    """Dirección de Fisher ponderada v_k de cada nodo k (ecuaciones de S_w y v).

    Xs : (n, q) variables sorteadas de cada observación (columnas propias de su nodo).
    Devuelve V (n_nodos, q) normalizadas y una máscara de nodos válidos (con al
    menos 2 observaciones de cada clase).
    """
    q = Xs.shape[1]
    w1, w0 = w * y, w * (1 - y)
    W1 = np.bincount(nodo, weights=w1, minlength=n_nodos)
    W0 = np.bincount(nodo, weights=w0, minlength=n_nodos)
    n1 = np.bincount(nodo, weights=(y == 1).astype(float), minlength=n_nodos)
    n0 = np.bincount(nodo, weights=(y == 0).astype(float), minlength=n_nodos)
    valido = (n1 >= 2) & (n0 >= 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu1 = np.stack([np.bincount(nodo, weights=w1 * Xs[:, j], minlength=n_nodos) for j in range(q)], 1) / W1[:, None]
        mu0 = np.stack([np.bincount(nodo, weights=w0 * Xs[:, j], minlength=n_nodos) for j in range(q)], 1) / W0[:, None]
    mu1, mu0 = np.nan_to_num(mu1), np.nan_to_num(mu0)
    D = Xs - np.where(y[:, None] == 1, mu1[nodo], mu0[nodo])
    WD = w[:, None] * D
    Sw = np.zeros((n_nodos, q, q))
    for a in range(q):
        for b in range(a, q):
            s = np.bincount(nodo, weights=WD[:, a] * D[:, b], minlength=n_nodos)
            Sw[:, a, b] = s
            Sw[:, b, a] = s
    Wtot = np.maximum(W1 + W0, 1e-12)
    Sw /= Wtot[:, None, None]
    traza = np.trace(Sw, axis1=1, axis2=2) / q
    Sw += (eps * traza + 1e-9)[:, None, None] * np.eye(q)[None]
    V = np.linalg.solve(Sw, (mu1 - mu0)[:, :, None])[:, :, 0]
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    return V, valido


# ---------------------------------------------------------------------------
# Un árbol
# ---------------------------------------------------------------------------
class _ArbolCSFOF:
    """Árbol oblicuo crecido por niveles. Cada nodo interno guarda las columnas
    que usa (``feat``, rellenas con -1), sus coeficientes (``coef``) y el umbral."""

    def fit(self, Z, Zb, bordes, y, w, cnt, q, qo, n_obl, max_depth, min_leaf, oblicuo, eps, n_bins, rng):
        """q: variables por nodo para los cortes por ejes; qo: variables de cada candidato
        oblicuo; n_obl: número de candidatos oblicuos por nodo."""
        n, p = Z.shape
        ancho = max(q, qo)                               # columnas guardadas por nodo
        filas = np.flatnonzero(cnt > 0)
        nodo_de = np.zeros(len(filas), dtype=np.int64)  # nodo (del nivel actual) de cada fila activa
        ids_nivel = np.array([0])                        # id global de cada nodo del nivel
        feat, coef, thr, izq, der, valor = [np.full((1, ancho), -1)], [np.zeros((1, ancho))], [0.0], [-1], [-1], [0.0]
        siguiente_id = 1

        for prof in range(max_depth + 1):
            K = len(ids_nivel)
            yf, wf, cf = y[filas], w[filas], cnt[filas]
            ww, wy = wf * cf, wf * cf * yf               # peso total y peso de morosos de cada fila
            W = np.bincount(nodo_de, weights=ww, minlength=K)
            W1 = np.bincount(nodo_de, weights=wy, minlength=K)
            C = np.bincount(nodo_de, weights=cf, minlength=K)
            for k in range(K):                           # valor de hoja provisional = p_hoja ponderado
                valor[ids_nivel[k]] = W1[k] / W[k] if W[k] > 0 else 0.0
            puede = (C >= 2 * min_leaf) & (W1 > 0) & (W1 < W) & (prof < max_depth)
            if not puede.any():
                break

            mejor_G = np.full(K, np.inf)
            FEAT = np.full((K, ancho), -1)               # corte elegido: columnas, coeficientes, umbral
            COEF = np.zeros((K, ancho))
            THR = np.zeros(K)

            # --- candidatos por ejes: q variables al azar por nodo ---
            F = np.argsort(rng.random((K, p)), axis=1)[:, :q]
            for j in range(q):
                col = F[nodo_de, j]
                G, b = _mejor_umbral_por_histograma(nodo_de, Zb[filas, col], ww, wy, cf, K, n_bins, min_leaf)
                m = np.flatnonzero(G < mejor_G)
                if len(m):
                    jj = F[m, j]
                    mejor_G[m] = G[m]
                    FEAT[m] = -1; COEF[m] = 0.0
                    FEAT[m, 0], COEF[m, 0] = jj, 1.0
                    THR[m] = [bordes[c][bb] for c, bb in zip(jj, b[m])]

            # --- candidatos oblicuos: Fisher ponderado, local al nodo ---
            if oblicuo and qo >= 2:
                inicio = np.concatenate([[0], np.cumsum(np.bincount(nodo_de, minlength=K))])
                tam = np.maximum(inicio[1:] - inicio[:-1], 1)
                for _ in range(n_obl):
                    Fo = np.argsort(rng.random((K, p)), axis=1)[:, :qo]
                    Xs = Z[filas[:, None], Fo[nodo_de]]
                    V, valido = direccion_fisher_por_nodo(Xs, yf, ww, nodo_de, K, eps)
                    z = np.einsum("ij,ij->i", Xs, V[nodo_de])
                    orden = np.lexsort((z, nodo_de))     # por nodo y, dentro, por z
                    rango = np.empty(len(z), dtype=np.int64)
                    rango[orden] = np.arange(len(z)) - inicio[nodo_de[orden]]
                    bz = np.minimum(rango * n_bins // tam[nodo_de], n_bins - 1)
                    G, b = _mejor_umbral_por_histograma(nodo_de, bz, ww, wy, cf, K, n_bins, min_leaf)
                    G[~valido] = np.inf
                    m = np.flatnonzero(G < mejor_G)
                    if len(m):                           # umbral: punto medio entre cuantiles vecinos
                        zs = z[orden]
                        nL = np.ceil((b[m] + 1) * tam[m] / n_bins).astype(np.int64)
                        s = inicio[m]
                        t_ = 0.5 * (zs[s + nL - 1] + zs[np.minimum(s + nL, inicio[m + 1] - 1)])
                        mejor_G[m] = G[m]
                        FEAT[m] = -1; COEF[m] = 0.0
                        FEAT[m, :qo], COEF[m, :qo], THR[m] = Fo[m], V[m], t_

            # --- aplicar cortes y crear hijos ---
            divide = puede & np.isfinite(mejor_G)
            if not divide.any():
                break
            ks = np.flatnonzero(divide)

            # Lado de cada fila: u^T x <= t (un corte por eje es u = e_j)
            f_fila = FEAT[nodo_de]
            proy = (np.where(f_fila >= 0, Z[filas[:, None], np.maximum(f_fila, 0)], 0.0) * COEF[nodo_de]).sum(1)
            va_izq = proy <= THR[nodo_de]

            # Registrar nodos internos e hijos
            nuevos_ids = siguiente_id + np.arange(2 * len(ks)).reshape(-1, 2)
            siguiente_id += 2 * len(ks)
            for r, k in enumerate(ks):
                gid = ids_nivel[k]
                feat[gid], coef[gid], thr[gid] = FEAT[k][None], COEF[k][None], float(THR[k])
                izq[gid], der[gid] = int(nuevos_ids[r, 0]), int(nuevos_ids[r, 1])
            n_nuevos = 2 * len(ks)
            feat.extend([np.full((1, ancho), -1)] * n_nuevos); coef.extend([np.zeros((1, ancho))] * n_nuevos)
            thr.extend([0.0] * n_nuevos); izq.extend([-1] * n_nuevos)
            der.extend([-1] * n_nuevos); valor.extend([0.0] * n_nuevos)

            # Filas que siguen activas y su nodo en el nivel siguiente
            sigue = divide[nodo_de]
            filas, nodo_de, va_izq = filas[sigue], nodo_de[sigue], va_izq[sigue]
            pos = np.full(K, -1)
            pos[ks] = np.arange(len(ks))
            nodo_de = 2 * pos[nodo_de] + np.where(va_izq, 0, 1)
            ids_nivel = nuevos_ids.ravel()

        # Tipos enteros compactos: el artefacto serializado pesa menos (Render
        # tiene memoria limitada) y las predicciones no cambian.
        self.feat_ = np.vstack(feat).astype(np.int16)
        self.coef_ = np.vstack(coef)
        self.thr_ = np.asarray(thr)
        self.izq_ = np.asarray(izq, dtype=np.int32)
        self.der_ = np.asarray(der, dtype=np.int32)
        self.valor_ = np.asarray(valor)
        return self

    def proba1(self, Z):
        nodo = np.zeros(len(Z), dtype=np.int64)
        activos = np.arange(len(Z))
        while len(activos):
            n_act = nodo[activos]
            interno = self.izq_[n_act] >= 0
            activos, n_act = activos[interno], n_act[interno]
            if not len(activos):
                break
            f = self.feat_[n_act]
            proy = (np.where(f >= 0, Z[activos[:, None], np.maximum(f, 0)], 0.0) * self.coef_[n_act]).sum(1)
            nodo[activos] = np.where(proy <= self.thr_[n_act], self.izq_[n_act], self.der_[n_act])
        return self.valor_[nodo]


def _ajustar_lote(semillas, Z, Zb, bordes, y, w, q, qo, n_obl, max_depth, min_leaf, oblicuo, eps, n_bins, bootstrap):
    """Ajusta varios árboles; cada uno usa su propia semilla (resultado independiente de n_jobs)."""
    n = len(y)
    arboles = []
    for s in semillas:
        rng = np.random.default_rng(int(s))
        cnt = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64) if bootstrap else np.ones(n)
        arboles.append(_ArbolCSFOF().fit(Z, Zb, bordes, y, w, cnt, q, qo, n_obl, max_depth, min_leaf, oblicuo, eps, n_bins, rng))
    return arboles


# ---------------------------------------------------------------------------
# El bosque
# ---------------------------------------------------------------------------
class CSFisherObliqueForest(ClassifierMixin, BaseEstimator):
    """Cost-Sensitive Fisher Oblique Forest.

    Parameters
    ----------
    n_estimators : int, número de árboles T.
    max_depth : int, profundidad máxima d.
    min_samples_leaf : int, mínimo de observaciones por hoja h.
    max_features : {"sqrt", "log2", None} | int | float, variables por nodo q.
    class_weight : {None, "balanced"} | dict, define el costo c del moroso.
    eps : float, regularización ε de la dispersión intra-clase.
    oblique : bool, activa el corte oblicuo (False = ablación «sin corte oblicuo»).
    oblique_features : None | "sqrt" | "log2" | int | float, variables de cada candidato
        oblicuo (None = las mismas q de los cortes por ejes).
    n_oblique : int, candidatos oblicuos por nodo, cada uno con su propio sorteo de variables.
    n_bins : int, número de cuantiles para la búsqueda de umbrales.
    bootstrap : bool, muestra bootstrap por árbol.
    random_state : int | None, semilla.
    n_jobs : int, procesos para ajustar los árboles (no cambia el resultado).
    """

    def __init__(self, n_estimators=200, max_depth=10, min_samples_leaf=20, max_features="sqrt",
                 class_weight=None, eps=0.1, oblique=True, oblique_features=None, n_oblique=1, n_bins=64, bootstrap=True, random_state=None, n_jobs=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.eps = eps
        self.oblique = oblique
        self.oblique_features = oblique_features
        self.n_oblique = n_oblique
        self.n_bins = n_bins
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, dtype=np.float64)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("CS-FOF es un clasificador binario: y debe tener exactamente 2 clases.")
        y01 = (y == self.classes_[1]).astype(np.float64)
        n, p = X.shape
        self.n_features_in_ = p

        w = _pesos_de_clase(y01, self.class_weight)
        if sample_weight is not None:
            w = w * np.asarray(sample_weight, dtype=np.float64)

        # Estandarización (estadísticos del entrenamiento) para que Fisher sea estable.
        self.media_ = X.mean(0)
        self.escala_ = X.std(0)
        self.escala_[self.escala_ == 0] = 1.0
        Z = (X - self.media_) / self.escala_

        # Bordes de cuantiles por variable y variables discretizadas (una vez por ajuste).
        cuant = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        bordes = []
        Zb = np.empty((n, p), dtype=np.int64)
        for j in range(p):
            e = np.unique(np.quantile(Z[:, j], cuant))
            e = np.append(e, np.inf)                   # el último bin incluye el máximo
            bordes.append(e)
            Zb[:, j] = np.searchsorted(e, Z[:, j], side="left")
        q = _n_features_por_nodo(self.max_features, p)
        rng = np.random.default_rng(self.random_state)
        semillas = rng.integers(0, 2**31 - 1, int(self.n_estimators))
        qo = q if self.oblique_features is None else _n_features_por_nodo(self.oblique_features, p)
        args = (Z, Zb, bordes, y01, w, q, max(2, qo), max(1, int(self.n_oblique)), int(self.max_depth), max(1, int(self.min_samples_leaf)),
                bool(self.oblique), float(self.eps), int(self.n_bins), bool(self.bootstrap))
        n_jobs = int(self.n_jobs or 1)
        if n_jobs == 1:
            self.arboles_ = _ajustar_lote(semillas, *args)
        else:
            lotes = [l for l in np.array_split(semillas, n_jobs) if len(l)]
            res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_ajustar_lote)(l, *args) for l in lotes)
            self.arboles_ = [a for lote in res for a in lote]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "arboles_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Se esperaban {self.n_features_in_} variables y llegaron {X.shape[1]}.")
        Z = (X - self.media_) / self.escala_
        p1 = np.mean([a.proba1(Z) for a in self.arboles_], axis=0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def fraccion_cortes_oblicuos(self) -> float:
        """Fracción de nodos internos que usan corte oblicuo (para el análisis de resultados)."""
        check_is_fitted(self, "arboles_")
        internos = oblicuos = 0
        for a in self.arboles_:
            m = a.izq_ >= 0
            internos += m.sum()
            oblicuos += ((a.feat_[m] >= 0).sum(1) > 1).sum()
        return oblicuos / max(internos, 1)
