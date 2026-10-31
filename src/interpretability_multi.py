"""Interpretabilidad con evidencia de más de un fold y más de una observación.

``interpretability.py`` explica UNA combinación en UN fold externo y compara
LIME con SHAP sobre UNA observación. Eso ilustra el método pero no permite
afirmar que la explicación sea estable ni que LIME y SHAP coincidan en
general. Este módulo agrega las dos cosas que faltaban:

1. **SHAP global por fold** (``shap_global_by_fold``): para cada uno de los 3
   folds externos se reajusta el pipeline con los ``best_params`` que el inner
   loop eligió en ESE fold, se calculan los valores SHAP sobre una muestra
   aleatoria del conjunto de prueba de ESE fold y se resume la importancia
   (media de |SHAP| por variable). ``importance_stability`` mide qué tan parecido
   queda el ranking entre folds (correlación de Spearman y solapamiento del
   top-k).
2. **LIME vs. SHAP sobre muchas observaciones** (``lime_vs_shap_agreement``):
   en cada fold se toma una muestra aleatoria de cada celda de la matriz de
   confusión (TP, TN, FP, FN), no solo los casos extremos, y para cada
   observación se mide cuánto coinciden los dos métodos en el orden de
   importancia, en el top-k y en el signo del efecto. Opcionalmente se repite
   LIME con distintas semillas para medir su propia variabilidad.

La comparación es de ORDEN y SIGNO, no de magnitud: el SHAP de XGBoost está en
escala de log-odds, el de LIME en escala de probabilidad.

Explicadores: ``TreeExplainer`` (exacto) para los modelos de árboles. No se
usan los explicadores model-agnostic de SHAP aquí: su costo por observación
es mucho mayor y la pregunta de este módulo es sobre los modelos que
lideran la tabla (Random Forest y XGBoost).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import config
from src import interpretability as ip
from src.models import MODEL_REGISTRY

TOP_K = 5


def _fold_artifacts(
    model_name: str,
    technique: str,
    params_per_fold: list[dict],
    X: pd.DataFrame,
    y: pd.Series,
    fold_index: int,
    n_outer_folds: int,
    random_state: int,
) -> dict[str, Any]:
    fitted = ip.fit_explainable_pipeline(
        MODEL_REGISTRY[model_name],
        technique,
        params_per_fold[fold_index],
        X,
        y,
        fold_index,
        n_outer_folds=n_outer_folds,
        random_state=random_state,
    )
    fitted["model_step"] = fitted["pipeline"].named_steps["model"]
    fitted["y_pred"] = fitted["pipeline"].predict(fitted["X_test"])
    fitted["y_proba"] = fitted["pipeline"].predict_proba(fitted["X_test"])[:, 1]
    return fitted


def shap_global_by_fold(
    model_name: str,
    technique: str,
    params_per_fold: list[dict],
    X: pd.DataFrame,
    y: pd.Series,
    n_outer_folds: int = config.N_OUTER_FOLDS,
    n_explain: int = 300,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Importancia global SHAP (media de |SHAP|) por variable y por fold externo.

    Las columnas one-hot de ``SEX``/``EDUCATION``/``MARRIAGE`` se suman de
    vuelta a su variable original (``ip.aggregate_onehot_shap``).

    Returns
    -------
    pd.DataFrame con columnas ``fold``, ``feature``, ``mean_abs_shap``,
    ``rank`` (1 = más importante dentro del fold) y ``n_explained``.
    """
    rows = []
    for fold_index in range(n_outer_folds):
        fitted = _fold_artifacts(
            model_name, technique, params_per_fold, X, y, fold_index, n_outer_folds, random_state
        )
        n = min(n_explain, fitted["X_test_t"].shape[0])
        rng = np.random.default_rng(random_state + fold_index)
        idx = np.sort(rng.choice(fitted["X_test_t"].shape[0], size=n, replace=False))
        shap_values, _ = ip.compute_shap_values(
            model_name, fitted["model_step"], fitted["X_train_t"], fitted["X_test_t"][idx]
        )
        agg_values, agg_names = ip.aggregate_onehot_shap(shap_values, fitted["feature_names"])
        importance = np.abs(agg_values).mean(axis=0)
        order = (-importance).argsort().argsort() + 1
        for name, value, rank in zip(agg_names, importance, order):
            rows.append(
                {"fold": fold_index, "feature": name, "mean_abs_shap": float(value),
                 "rank": int(rank), "n_explained": n}
            )
    return pd.DataFrame(rows)


def importance_stability(importance: pd.DataFrame, top_k: int = TOP_K) -> dict[str, Any]:
    """Estabilidad del ranking de importancia SHAP entre folds.

    Returns
    -------
    dict con ``spearman`` (DataFrame fold x fold), ``spearman_mean_offdiag``,
    ``topk_jaccard`` (DataFrame fold x fold) y ``topk_always`` (variables que
    están en el top-k de TODOS los folds).
    """
    pivot = importance.pivot(index="feature", columns="fold", values="mean_abs_shap")
    folds = list(pivot.columns)
    spear = pivot.corr(method="spearman")
    tops = {f: set(pivot[f].nlargest(top_k).index) for f in folds}
    jac = pd.DataFrame(
        [[len(tops[a] & tops[b]) / len(tops[a] | tops[b]) for b in folds] for a in folds],
        index=folds,
        columns=folds,
    )
    off = ~np.eye(len(folds), dtype=bool)
    return {
        "spearman": spear,
        "spearman_mean_offdiag": float(spear.to_numpy()[off].mean()),
        "topk_jaccard": jac,
        "topk_jaccard_mean_offdiag": float(jac.to_numpy()[off].mean()),
        "topk_always": sorted(set.intersection(*tops.values())),
    }


def sample_observations_by_outcome(
    y_true: np.ndarray, y_pred: np.ndarray, per_cell: int, random_state: int
) -> pd.DataFrame:
    """Muestra aleatoria (no solo extremos) de cada celda de la matriz de
    confusión: ``TN``, ``FP``, ``FN``, ``TP``. Devuelve ``position`` y ``outcome``."""
    rng = np.random.default_rng(random_state)
    cells = {
        "TN": (y_true == 0) & (y_pred == 0),
        "FP": (y_true == 0) & (y_pred == 1),
        "FN": (y_true == 1) & (y_pred == 0),
        "TP": (y_true == 1) & (y_pred == 1),
    }
    rows = []
    for outcome, mask in cells.items():
        positions = np.where(mask)[0]
        chosen = rng.choice(positions, size=min(per_cell, len(positions)), replace=False)
        rows.extend({"position": int(p), "outcome": outcome} for p in chosen)
    return pd.DataFrame(rows)


def _agreement_metrics(shap_row: np.ndarray, lime_weights: dict[int, float], top_k: int) -> dict[str, float]:
    """Coincidencia entre un vector SHAP y los pesos LIME de la MISMA observación.

    LIME solo entrega ``num_features`` pesos; el resto se toma como 0. La
    correlación de Spearman se calcula sobre |atribución| de todas las
    variables, el solapamiento sobre el top-k de cada método y el signo sobre
    las variables del top-k de SHAP que LIME también reporta.
    """
    n_features = len(shap_row)
    lime_vec = np.zeros(n_features)
    for idx, weight in lime_weights.items():
        lime_vec[idx] = weight

    rho = spearmanr(np.abs(shap_row), np.abs(lime_vec)).statistic
    shap_top = set(np.argsort(-np.abs(shap_row))[:top_k])
    lime_top = set(sorted(lime_weights, key=lambda i: -abs(lime_weights[i]))[:top_k])
    overlap = len(shap_top & lime_top) / top_k

    shared = [i for i in shap_top if i in lime_weights and shap_row[i] != 0]
    sign_agree = (
        float(np.mean([np.sign(shap_row[i]) == np.sign(lime_weights[i]) for i in shared]))
        if shared
        else np.nan
    )
    return {"spearman_abs": float(rho), f"top{top_k}_overlap": overlap, "sign_agreement_topk": sign_agree}


def lime_vs_shap_agreement(
    model_name: str,
    technique: str,
    params_per_fold: list[dict],
    X: pd.DataFrame,
    y: pd.Series,
    n_outer_folds: int = config.N_OUTER_FOLDS,
    per_cell: int = 8,
    num_features: int = 10,
    top_k: int = TOP_K,
    lime_seeds: tuple[int, ...] = (config.RANDOM_STATE,),
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """LIME vs. SHAP sobre ``4 * per_cell`` observaciones por fold externo.

    Con ``lime_seeds`` de más de un valor, LIME se repite con cada semilla y
    se agrega ``lime_seed_topk_overlap``: solapamiento medio del top-k entre
    pares de semillas (la variabilidad propia de LIME, sin SHAP de por medio).

    Returns
    -------
    pd.DataFrame, una fila por observación, con ``fold``, ``outcome``,
    ``position``, ``proba`` y las métricas de coincidencia.
    """
    rows = []
    for fold_index in range(n_outer_folds):
        fitted = _fold_artifacts(
            model_name, technique, params_per_fold, X, y, fold_index, n_outer_folds, random_state
        )
        sample = sample_observations_by_outcome(
            fitted["y_test"].to_numpy(), fitted["y_pred"], per_cell, random_state + 100 + fold_index
        )
        positions = sample["position"].to_numpy()
        shap_values, _ = ip.compute_shap_values(
            model_name, fitted["model_step"], fitted["X_train_t"], fitted["X_test_t"][positions]
        )
        for (_, obs), shap_row in zip(sample.iterrows(), shap_values):
            pos = int(obs["position"])
            x_row = fitted["X_test_t"][pos : pos + 1]
            lime_runs = []
            for seed in lime_seeds:
                exp = ip.explain_with_lime(
                    fitted["model_step"], fitted["X_train_t"], fitted["feature_names"], x_row,
                    num_features=num_features, random_state=seed,
                )
                lime_runs.append({i: w for i, w in exp.as_map()[1]})
            record = {
                "model": model_name, "technique": technique, "fold": fold_index,
                "outcome": obs["outcome"], "position": pos, "proba": float(fitted["y_proba"][pos]),
                **_agreement_metrics(shap_row, lime_runs[0], top_k),
            }
            if len(lime_runs) > 1:
                tops = [set(sorted(r, key=lambda i: -abs(r[i]))[:top_k]) for r in lime_runs]
                pair_overlaps = [
                    len(tops[a] & tops[b]) / top_k
                    for a in range(len(tops))
                    for b in range(a + 1, len(tops))
                ]
                record["lime_seed_topk_overlap"] = float(np.mean(pair_overlaps))
            rows.append(record)
    return pd.DataFrame(rows)


def summarize_agreement(agreement: pd.DataFrame) -> pd.DataFrame:
    """Media, desviación estándar y mediana de cada métrica, por celda de
    la matriz de confusión y en total."""
    metrics = [c for c in agreement.columns if c in ("spearman_abs", f"top{TOP_K}_overlap",
                                                     "sign_agreement_topk", "lime_seed_topk_overlap")]
    stats = ["mean", "std", "median"]
    by_outcome = agreement.groupby("outcome")[metrics].agg(stats)
    overall = agreement.assign(_g="ALL").groupby("_g")[metrics].agg(stats)
    overall.index = ["ALL"]
    return pd.concat([by_outcome, overall])


INTERPRETABILITY_DIR = config.RESULTS_DIR / "interpretability"


def run_for_best_models(
    models: tuple[str, ...] = ("random_forest", "xgboost"),
    n_explain: int = 300,
    per_cell: int = 10,
    lime_seeds: tuple[int, ...] = (42, 43, 44),
) -> None:
    """Corre SHAP por fold y LIME vs. SHAP para la mejor combinación de cada
    modelo de ``models`` (según ``results/best_per_model.csv``) y guarda los
    CSV en ``results/interpretability/``."""
    import json

    from src.data_loader import load_processed_dataset, split_features_target
    from src.preprocessing import remove_duplicate_rows

    df, _ = remove_duplicate_rows(load_processed_dataset())
    X, y = split_features_target(df)
    best = pd.read_csv(config.RESULTS_DIR / "best_per_model.csv")
    master = pd.read_csv(config.RESULTS_DIR / "master_table_112.csv")
    INTERPRETABILITY_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        b = best[best["model"] == model_name].iloc[0]
        row = master[
            (master["model"] == model_name)
            & (master["technique"] == b["technique"])
            & (master["method"] == b["method"])
        ].iloc[0]
        params = json.loads(row["best_params_per_fold"])
        n_outer = int(row["n_outer_folds"])

        importance = shap_global_by_fold(
            model_name, b["technique"], params, X, y, n_outer_folds=n_outer, n_explain=n_explain
        )
        importance.insert(0, "model", model_name)
        importance.to_csv(INTERPRETABILITY_DIR / f"shap_importance_by_fold_{model_name}.csv", index=False)

        agreement = lime_vs_shap_agreement(
            model_name, b["technique"], params, X, y, n_outer_folds=n_outer,
            per_cell=per_cell, lime_seeds=lime_seeds,
        )
        agreement.to_csv(INTERPRETABILITY_DIR / f"lime_vs_shap_{model_name}.csv", index=False)
        print(f"{model_name}: {len(importance)} filas de importancia, {len(agreement)} observaciones", flush=True)


if __name__ == "__main__":
    run_for_best_models()
