"""Modelos neuronales para datos tabulares, compatibles con scikit-learn.

- ``TabMClassifier``: TabM (Gorishniy et al., ICLR 2025), un ensamble eficiente de k MLP que comparten
  pesos (BatchEnsemble). Pérdida de entropía cruzada **ponderada por el costo** (los mismos pesos
  ``w_i`` de CS-FOF) y parada temprana con una fracción de validación tomada solo del entrenamiento.
- **Modelo original, TabM-Fisher** (``TabMClassifier(fisher_features=m)``): antes de entrenar la red se
  ajusta un CS-FOF poco profundo con el mismo entrenamiento y se extraen sus cortes oblicuos de Fisher
  de los primeros niveles. Cada corte ``(v, t)`` se convierte en una variable nueva, la distancia con
  signo ``v^T x - t`` del cliente a la frontera. La red recibe las variables originales más esas ``m``
  distancias. Así TabM parte de las fronteras diagonales que CS-FOF ya encontró entre los atrasos
  correlacionados y dedica su capacidad a las señales débiles y no lineales (los clientes al día, donde
  CS-FOF pierde frente a XGBoost). La ablación es ``fisher_features=0`` (TabM simple).
- ``TabPFNClasificador``: envoltorio de TabPFN (Hollmann et al., 2025) con la versión elegida (2.5 por
  defecto, la que propone el curso). No se entrena: aprende en contexto, así que no tiene búsqueda de
  hiperparámetros. ``class_weight="balanced"`` se traduce a ``balance_probabilities=True``.

Las dependencias (``torch``, ``tabm``, ``tabpfn``) se importan solo al ajustar, para que el resto del
proyecto funcione sin ellas.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def _pesos_costo(y01: np.ndarray, class_weight) -> tuple[float, float]:
    """Pesos (w0, w1) de cada clase. "balanced" -> w1 = n0/n1 y w0 = 1, como en CS-FOF."""
    if class_weight is None:
        return 1.0, 1.0
    if class_weight == "balanced":
        n1 = max(y01.sum(), 1.0)
        return 1.0, float((len(y01) - y01.sum()) / n1)
    if isinstance(class_weight, dict):
        return float(class_weight.get(0, 1.0)), float(class_weight.get(1, 1.0))
    raise ValueError(f"class_weight no reconocido: {class_weight!r}")


def _dispositivo(device: str):
    import torch
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def cortes_fisher(bosque, max_prof: int = 2, max_cortes: int | None = None) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Cortes oblicuos ``(columnas, coeficientes, umbral)`` de los primeros niveles de un CS-FOF ajustado.

    Recorre cada árbol por niveles desde la raíz (profundidad 0) hasta ``max_prof`` y se queda con los
    nodos internos cuyo corte usa más de una variable (corte oblicuo de Fisher).
    """
    check_is_fitted(bosque, "arboles_")
    cortes = []
    for a in bosque.arboles_:
        nivel, prof = [0], 0
        while nivel and prof <= max_prof:
            siguiente = []
            for nodo in nivel:
                if a.izq_[nodo] < 0:
                    continue
                cols = a.feat_[nodo]
                m = cols >= 0
                if m.sum() > 1:
                    cortes.append((cols[m].astype(int), a.coef_[nodo][m].astype(float), float(a.thr_[nodo])))
                siguiente += [int(a.izq_[nodo]), int(a.der_[nodo])]
            nivel, prof = siguiente, prof + 1
    return cortes[:max_cortes] if max_cortes else cortes


class TabMClassifier(ClassifierMixin, BaseEstimator):
    """TabM sensible al costo, con variables de Fisher opcionales (TabM-Fisher).

    Parameters
    ----------
    k : int, número de miembros del ensamble TabM.
    n_blocks, d_block, dropout : arquitectura del MLP de cada miembro.
    lr, weight_decay : AdamW.
    batch_size, max_epochs, patience : entrenamiento con parada temprana (log-loss ponderada de validación).
    val_fraction : fracción del entrenamiento (estratificada) reservada para la parada temprana.
    class_weight : None | "balanced" | dict, costo del moroso en la pérdida.
    fisher_features : int, número de cortes de Fisher que se agregan como variables (0 = TabM simple).
    fisher_trees, fisher_depth : tamaño del CS-FOF auxiliar (árboles y profundidad de la que se toman cortes).
    random_state : semilla.
    device : "auto" | "cpu" | "cuda".
    """

    def __init__(self, k=32, n_blocks=3, d_block=512, dropout=0.1, lr=2e-3, weight_decay=3e-4, batch_size=256,
                 max_epochs=100, patience=10, val_fraction=0.1, class_weight=None, fisher_features=0,
                 fisher_trees=50, fisher_depth=2, random_state=None, device="auto"):
        self.k = k
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.class_weight = class_weight
        self.fisher_features = fisher_features
        self.fisher_trees = fisher_trees
        self.fisher_depth = fisher_depth
        self.random_state = random_state
        self.device = device

    # --- variables de Fisher -------------------------------------------------------------
    def _ajustar_fisher(self, X, y01):
        from src.original_model import CSFisherObliqueForest
        self.bosque_ = CSFisherObliqueForest(
            n_estimators=int(self.fisher_trees), max_depth=int(self.fisher_depth) + 1, min_samples_leaf=50,
            max_features="sqrt", class_weight=self.class_weight, random_state=self.random_state, n_jobs=1,
        ).fit(X, y01)
        self.cortes_ = cortes_fisher(self.bosque_, int(self.fisher_depth), int(self.fisher_features))
        F = self._distancias(X)
        self.f_media_, self.f_escala_ = F.mean(0), F.std(0)
        self.f_escala_[self.f_escala_ == 0] = 1.0

    def _distancias(self, X):
        Z = (X - self.bosque_.media_) / self.bosque_.escala_
        if not self.cortes_:
            return np.zeros((len(X), 0))
        return np.column_stack([Z[:, c] @ v - t for c, v, t in self.cortes_])

    def _entrada(self, X):
        if not self.fisher_features:
            return X
        return np.hstack([X, (self._distancias(X) - self.f_media_) / self.f_escala_])

    # --- entrenamiento ---------------------------------------------------------------------
    def fit(self, X, y):
        import torch
        from tabm import TabM

        X, y = check_X_y(X, y, dtype=np.float64)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("TabMClassifier es binario: y debe tener exactamente 2 clases.")
        y01 = (y == self.classes_[1]).astype(np.int64)
        self.n_features_in_ = X.shape[1]
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(int(rng.randint(0, 2**31 - 1)))

        idx_tr, idx_va = train_test_split(np.arange(len(y01)), test_size=self.val_fraction, stratify=y01,
                                          random_state=rng.randint(0, 2**31 - 1))
        if self.fisher_features:
            self._ajustar_fisher(X[idx_tr], y01[idx_tr])  # solo con la parte de entrenamiento
        A = self._entrada(X).astype(np.float32)
        w0, w1 = _pesos_costo(y01[idx_tr].astype(float), self.class_weight)
        dev = _dispositivo(self.device)
        pesos = torch.tensor([w0, w1], dtype=torch.float32, device=dev)

        self.red_ = TabM.make(n_num_features=A.shape[1], d_out=2, k=int(self.k), n_blocks=int(self.n_blocks),
                              d_block=int(self.d_block), dropout=float(self.dropout)).to(dev)
        opt = torch.optim.AdamW(self.red_.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))
        Xtr, ytr = torch.tensor(A[idx_tr], device=dev), torch.tensor(y01[idx_tr], device=dev)
        Xva, yva = torch.tensor(A[idx_va], device=dev), torch.tensor(y01[idx_va], device=dev)
        gen = torch.Generator(device="cpu").manual_seed(int(rng.randint(0, 2**31 - 1)))

        mejor, sin_mejora, mejor_estado, self.historial_ = np.inf, 0, None, []
        for epoca in range(int(self.max_epochs)):
            self.red_.train()
            for lote in torch.randperm(len(ytr), generator=gen).split(int(self.batch_size)):
                lote = lote.to(dev)
                logits = self.red_(Xtr[lote])                       # (B, k, 2)
                # Cada miembro se entrena con su propia pérdida (receta de TabM).
                perdida = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), ytr[lote].repeat_interleave(logits.shape[1]), weight=pesos)
                opt.zero_grad()
                perdida.backward()
                opt.step()
            val = self._logloss(Xva, yva, pesos)
            self.historial_.append(val)
            if val < mejor - 1e-5:
                mejor, sin_mejora = val, 0
                mejor_estado = {n: t.detach().clone() for n, t in self.red_.state_dict().items()}
            else:
                sin_mejora += 1
                if sin_mejora >= int(self.patience):
                    break
        self.red_.load_state_dict(mejor_estado)
        self.n_epocas_ = epoca + 1
        self.red_.cpu()
        return self

    def _proba_tensor(self, Xt):
        import torch
        self.red_.eval()
        salidas = []
        with torch.no_grad():
            for lote in Xt.split(4096):
                salidas.append(torch.softmax(self.red_(lote), dim=-1).mean(1))  # promedio de los k miembros
        return torch.cat(salidas)

    def _logloss(self, Xva, yva, pesos):
        import torch
        p = self._proba_tensor(Xva).clamp(1e-7, 1 - 1e-7)
        w = pesos[yva]
        return float(-(w * torch.log(p[torch.arange(len(yva)), yva])).sum() / w.sum())

    def predict_proba(self, X):
        import torch
        check_is_fitted(self, "red_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Se esperaban {self.n_features_in_} variables y llegaron {X.shape[1]}.")
        return self._proba_tensor(torch.tensor(self._entrada(X).astype(np.float32))).numpy().astype(np.float64)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class TabPFNClasificador(ClassifierMixin, BaseEstimator):
    """TabPFN (aprendizaje en contexto) con la versión de pesos elegida.

    Requiere aceptar la licencia del modelo en Hugging Face e iniciar sesión (``HF_TOKEN``).
    ``max_contexto`` limita las filas de entrenamiento (submuestra estratificada) por memoria de la GPU.
    """

    def __init__(self, version="v2.5", n_estimators=8, max_contexto=50_000, class_weight=None,
                 random_state=None, device="auto"):
        self.version = version
        self.n_estimators = n_estimators
        self.max_contexto = max_contexto
        self.class_weight = class_weight
        self.random_state = random_state
        self.device = device

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        X, y = check_X_y(X, y, dtype=np.float64)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        if len(y) > self.max_contexto:
            sel, _ = train_test_split(np.arange(len(y)), train_size=int(self.max_contexto), stratify=y,
                                      random_state=self.random_state)
            X, y = X[sel], y[sel]
        dispositivo = str(_dispositivo(self.device))
        self.modelo_ = TabPFNClassifier.create_default_for_version(
            ModelVersion(self.version), n_estimators=int(self.n_estimators),
            balance_probabilities=self.class_weight == "balanced", random_state=self.random_state,
            device=dispositivo, ignore_pretraining_limits=True,
        )
        self.modelo_.fit(X, y)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "modelo_")
        return self.modelo_.predict_proba(check_array(X, dtype=np.float64))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
