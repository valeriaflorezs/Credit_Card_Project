"""Los 4 métodos de optimización de hiperparámetros: Grid Search, Random
Search, Optimización Bayesiana (Optuna) y Algoritmos Genéticos (DEAP).

Todos comparten la misma fuente de verdad para el espacio de búsqueda
(``models.ModelSpec.hyperparameters``, ver ``models.py``) y la misma forma
de construir el pipeline preprocesamiento+balanceo+modelo
(``balancing.build_pipeline``, ver ``balancing.py``), y todos devuelven un
``OptimizationResult`` con la misma forma — así ``nested_cv.py`` (bloque 5)
y ``experiment_runner.py`` (bloque 6) pueden tratarlos de forma intercambiable
vía ``run_optimization(method, ...)``.

**Grid Search y Random Search** delegan la partición en folds internos a
``GridSearchCV``/``RandomizedSearchCV`` de scikit-learn (que clona y
reajusta el pipeline internamente por fold — sin leakage, ver
``balancing.py``).

**Bayesiana y Genética** implementan manualmente el pipeline completo
dentro de cada iteración de validación (requisito del curso): cada trial
de Optuna / cada individuo de DEAP construye su propio pipeline con
``balancing.build_pipeline`` y lo evalúa con su propio bucle de CV interno
(``cv.split``), en vez de delegarlo a un ``SearchCV`` de scikit-learn.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from deap import base, creator, tools
from joblib import Parallel, delayed
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src import config
from src.balancing import build_pipeline
from src.models import ModelSpec, get_param_distributions, get_param_grid

optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTIMIZATION_METHODS: tuple[str, ...] = (
    "grid_search",
    "random_search",
    "bayesian_optuna",
    "genetic_deap",
)


@dataclass
class OptimizationResult:
    """Resultado uniforme de cualquiera de los 4 métodos de optimización.

    Attributes
    ----------
    method : str
        Uno de ``OPTIMIZATION_METHODS``.
    best_params : dict
        Mejores hiperparámetros encontrados (sin prefijo de pipeline).
    best_score : float
        Score promedio de validación INTERNA (inner loop) del mejor
        candidato — nunca debe reportarse como métrica final; esa siempre
        proviene de evaluar ``best_pipeline`` sobre el fold externo (outer
        loop, ver ``nested_cv.py``).
    best_pipeline : imblearn.pipeline.Pipeline
        Pipeline completo (preprocesamiento + balanceo + modelo) ya
        reajustado (``.fit()``) con ``best_params`` sobre TODO el conjunto
        ``(X, y)`` recibido por el optimizador (típicamente el fold de
        entrenamiento externo completo).
    n_evaluations : int
        Número de configuraciones de hiperparámetros efectivamente
        evaluadas (para comparar costo computacional entre métodos).
    diagnostics : dict
        Información específica del método (ej. sampler/acquisition en
        Optuna; operadores genéticos e historial de diversidad en DEAP).
    """

    method: str
    best_params: dict[str, Any]
    best_score: float
    best_pipeline: Any
    n_evaluations: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _n_splits(cv: Any) -> int:
    n = getattr(cv, "n_splits", None)
    if n is None:
        raise ValueError("El splitter de CV provisto no expone 'n_splits'.")
    return int(n)


# ---------------------------------------------------------------------------
# Grid Search / Random Search — delegan el bucle de CV interno a scikit-learn
# ---------------------------------------------------------------------------


def run_grid_search(
    spec: ModelSpec,
    technique: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    scoring: str = config.DEFAULT_SCORING,
    n_jobs: int = 1,
    random_state: int = config.RANDOM_STATE,
    preprocessing_steps: list[tuple[str, Any]] | None = None,
) -> OptimizationResult:
    """Grid Search exhaustivo sobre el espacio de búsqueda del modelo.

    Parameters
    ----------
    spec : ModelSpec
        Modelo base (ver ``models.MODEL_REGISTRY``).
    technique : str
        Técnica de balanceo (ver ``balancing.BALANCING_TECHNIQUES``).
    X, y : DataFrame, Series
        Fold de entrenamiento EXTERNO completo (outer loop); ``cv`` lo
        particiona internamente en los folds del inner loop.
    cv : sklearn CV splitter
        Ej. ``StratifiedKFold(n_splits=config.N_INNER_FOLDS, shuffle=True,
        random_state=...)``.
    scoring : str, default=config.DEFAULT_SCORING
        Nombre de scorer de scikit-learn (ver ``sklearn.metrics.get_scorer``).
    n_jobs : int, default=1
        Se deja en 1 por defecto para no anidar paralelismo: el paralelismo
        del pipeline completo se maneja a nivel de ``experiment_runner.py``
        (bloque 6), que ya paraleliza sobre las 112 combinaciones.
    random_state : int, default=config.RANDOM_STATE
    preprocessing_steps : list of (str, estimator), optional
        Ver ``balancing.build_pipeline`` (típicamente
        ``preprocessing.get_preprocessing_steps(spec)``).

    Returns
    -------
    OptimizationResult
    """
    base_pipeline = build_pipeline(
        spec,
        technique,
        y_train=y,
        cv_n_splits=_n_splits(cv),
        preprocessing_steps=preprocessing_steps,
        random_state=random_state,
    )
    param_grid = {
        f"model__{k}": v
        for k, v in get_param_grid(spec, max_combinations=spec.grid_search_max_combinations).items()
    }

    search = GridSearchCV(
        base_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X, y)

    return OptimizationResult(
        method="grid_search",
        best_params={k.replace("model__", ""): v for k, v in search.best_params_.items()},
        best_score=float(search.best_score_),
        best_pipeline=search.best_estimator_,
        n_evaluations=len(search.cv_results_["params"]),
        diagnostics={"param_grid": param_grid},
    )


def run_random_search(
    spec: ModelSpec,
    technique: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    scoring: str = config.DEFAULT_SCORING,
    n_iter: int = config.RANDOM_SEARCH_ITER,
    n_jobs: int = 1,
    random_state: int = config.RANDOM_STATE,
    preprocessing_steps: list[tuple[str, Any]] | None = None,
) -> OptimizationResult:
    """Random Search sobre el espacio de búsqueda del modelo.

    Ver ``run_grid_search`` para la semántica compartida de parámetros.
    Se prioriza sobre Grid Search para espacios amplios (ej. Random Forest,
    XGBoost), por recomendación explícita del curso dado el volumen de
    112 combinaciones.

    Parameters
    ----------
    n_iter : int, default=config.RANDOM_SEARCH_ITER
        Número de configuraciones muestreadas aleatoriamente.
    preprocessing_steps : list of (str, estimator), optional
        Ver ``run_grid_search``.
    """
    base_pipeline = build_pipeline(
        spec,
        technique,
        y_train=y,
        cv_n_splits=_n_splits(cv),
        preprocessing_steps=preprocessing_steps,
        random_state=random_state,
    )
    param_distributions = {f"model__{k}": v for k, v in get_param_distributions(spec).items()}

    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        refit=True,
    )
    search.fit(X, y)

    return OptimizationResult(
        method="random_search",
        best_params={k.replace("model__", ""): v for k, v in search.best_params_.items()},
        best_score=float(search.best_score_),
        best_pipeline=search.best_estimator_,
        n_evaluations=len(search.cv_results_["params"]),
        diagnostics={"n_iter": n_iter},
    )


# ---------------------------------------------------------------------------
# Optimización Bayesiana (Optuna) — pipeline manual dentro de cada trial
# ---------------------------------------------------------------------------


def _suggest_from_spec(trial: "optuna.Trial", spec: ModelSpec) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for hp in spec.hyperparameters:
        if hp.param_type == "categorical":
            params[hp.name] = trial.suggest_categorical(hp.name, hp.choices)
        elif hp.param_type == "int":
            params[hp.name] = trial.suggest_int(hp.name, hp.low, hp.high, log=hp.log)
        elif hp.param_type == "float":
            params[hp.name] = trial.suggest_float(hp.name, hp.low, hp.high, log=hp.log)
    return params


def run_bayesian_search(
    spec: ModelSpec,
    technique: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    scoring: str = config.DEFAULT_SCORING,
    n_trials: int = config.OPTUNA_N_TRIALS,
    random_state: int = config.RANDOM_STATE,
    preprocessing_steps: list[tuple[str, Any]] | None = None,
    n_jobs: int = 1,
) -> OptimizationResult:
    """Optimización Bayesiana con Optuna, pipeline construido manualmente por trial.

    Modelo sustituto (surrogate)
    -----------------------------
    Optuna usa por defecto ``TPESampler`` (Tree-structured Parzen Estimator),
    NO un proceso gaussiano. En vez de modelar directamente
    ``p(score | hiperparámetros)`` con un GP, TPE modela dos densidades no
    paramétricas sobre los hiperparámetros ya evaluados: ``l(x)`` (los que
    dieron buen score, el mejor cuantil ``gamma``) y ``g(x)`` (el resto).
    Se eligió TPE (el default de Optuna) porque escala mejor que un GP en
    espacios de búsqueda de dimensión moderada con parámetros mixtos
    (continuos + categóricos, como aquí), sin asumir suavidad/continuidad
    global de la función objetivo.

    Función de adquisición
    -----------------------
    TPE elige el siguiente punto maximizando la razón ``l(x) / g(x)``, lo
    cual es equivalente (Bergstra et al., 2011) a maximizar el Expected
    Improvement (EI). Esto privilegia la EXPLOTACIÓN de regiones ya
    prometedoras, moderado por: (a) una fase inicial de ``n_startup_trials``
    puramente aleatoria (exploración pura antes de ajustar las densidades),
    y (b) el propio muestreo probabilístico de ``l(x)``, que no colapsa a
    un único óptimo puntual.

    Parameters
    ----------
    n_trials : int, default=config.OPTUNA_N_TRIALS
    n_jobs : int, default=1
        Trials en paralelo (hilos, soporte nativo de Optuna vía
        ``study.optimize``). No cambia el algoritmo TPE en si ni el numero
        de trials — solo la concurrencia con la que se ejecutan. Con
        ``n_jobs>1`` puede haber hasta ``n_jobs`` trials "en vuelo"
        simultaneamente evaluados contra el mismo estado previo del study
        (en vez de estrictamente uno-a-la-vez); es el comportamiento
        soportado y documentado de Optuna para paralelizar, no un cambio
        metodologico.

    Returns
    -------
    OptimizationResult
    """
    scorer = get_scorer(scoring)
    n_splits = _n_splits(cv)

    def objective(trial: "optuna.Trial") -> float:
        params = _suggest_from_spec(trial, spec)
        fold_scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            pipe = build_pipeline(
                spec,
                technique,
                y_train=y_tr,
                preprocessing_steps=preprocessing_steps,
                random_state=random_state,
                **params,
            )
            pipe.fit(X_tr, y_tr)
            fold_scores.append(scorer(pipe, X_val, y_val))
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)

    best_params = study.best_params
    best_pipeline = build_pipeline(
        spec,
        technique,
        y_train=y,
        cv_n_splits=n_splits,
        preprocessing_steps=preprocessing_steps,
        random_state=random_state,
        **best_params,
    )
    best_pipeline.fit(X, y)

    return OptimizationResult(
        method="bayesian_optuna",
        best_params=best_params,
        best_score=float(study.best_value),
        best_pipeline=best_pipeline,
        n_evaluations=len(study.trials),
        diagnostics={
            "surrogate": "TPE (Tree-structured Parzen Estimator)",
            "acquisition": "Expected Improvement (EI), via razon de densidades l(x)/g(x)",
            "trade_off": "explotacion de regiones prometedoras, con arranque aleatorio inicial",
            "trials_dataframe": study.trials_dataframe(),
        },
    )


# ---------------------------------------------------------------------------
# Algoritmos Genéticos (DEAP) — pipeline manual dentro de cada evaluación
# ---------------------------------------------------------------------------


def _init_gene(hp) -> Any:
    if hp.param_type == "categorical":
        return random.choice(hp.choices)
    if hp.param_type == "int":
        return random.randint(hp.low, hp.high)
    # float
    if hp.log:
        return float(10 ** random.uniform(np.log10(hp.low), np.log10(hp.high)))
    return random.uniform(hp.low, hp.high)


def _cx_uniform(ind1: list, ind2: list, indpb: float = 0.5) -> tuple[list, list]:
    """Crossover uniforme: cada gen se intercambia independientemente con
    probabilidad ``indpb``. Se prefiere sobre un punto de corte único
    (``cxOnePoint``) porque los genes representan hiperparámetros
    heterogéneos sin una noción natural de "adyacencia" a preservar."""
    for i in range(len(ind1)):
        if random.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


def _mutate_individual(
    individual: list, hyperparams: list, indpb: float, sigma: float = 0.15
) -> tuple[list]:
    """Mutación por gen, consciente del tipo de hiperparámetro:

    - categorical: reasigna a una categoría aleatoria (uniforme).
    - int/float sin log: ruido gaussiano proporcional al rango, clip a límites.
    - int/float con log: ruido gaussiano en escala log10, clip a límites
      (para no romper la búsqueda logarítmica justificada en ``models.py``).
    """
    for i, hp in enumerate(hyperparams):
        if random.random() > indpb:
            continue
        if hp.param_type == "categorical":
            individual[i] = random.choice(hp.choices)
            continue
        lo, hi = hp.low, hp.high
        if hp.log:
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            current_log = np.log10(max(individual[i], 1e-300))
            new_log = current_log + random.gauss(0, sigma * (log_hi - log_lo))
            new_log = min(max(new_log, log_lo), log_hi)
            new_value = 10**new_log
        else:
            new_value = individual[i] + random.gauss(0, sigma * (hi - lo))
            new_value = min(max(new_value, lo), hi)
        individual[i] = int(round(new_value)) if hp.param_type == "int" else float(new_value)
    return (individual,)


def _normalize_gene(value: Any, hp) -> float:
    lo, hi = hp.low, hp.high
    if hp.log:
        v, lo_, hi_ = np.log10(max(value, 1e-300)), np.log10(lo), np.log10(hi)
    else:
        v, lo_, hi_ = value, lo, hi
    return 0.0 if hi_ == lo_ else (v - lo_) / (hi_ - lo_)


def population_diversity(population: list, hyperparams: list) -> float:
    """Índice de diversidad genética de la población en (aprox.) [0, 1].

    Para cada hiperparámetro: desviación estándar normalizada de sus
    valores en la población (numéricos) o fracción de valores distintos
    (categóricos); se promedia sobre todos los hiperparámetros. Un valor
    que colapsa hacia 0 a través de las generaciones es indicador de
    convergencia prematura (la población dejó de explorar).
    """
    per_hp_diversity = []
    for i, hp in enumerate(hyperparams):
        values = [ind[i] for ind in population]
        if hp.param_type == "categorical":
            per_hp_diversity.append(len(set(values)) / len(values))
        else:
            per_hp_diversity.append(float(np.std([_normalize_gene(v, hp) for v in values])))
    return float(np.mean(per_hp_diversity))


def run_genetic_search(
    spec: ModelSpec,
    technique: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    scoring: str = config.DEFAULT_SCORING,
    population_size: int = config.GA_POPULATION_SIZE,
    n_generations: int = config.GA_N_GENERATIONS,
    cx_prob: float = config.GA_CX_PROB,
    mut_prob: float = config.GA_MUT_PROB,
    mut_indpb: float = config.GA_MUT_INDPB,
    tournament_size: int = config.GA_TOURNAMENT_SIZE,
    elite_size: int = config.GA_ELITE_SIZE,
    random_state: int = config.RANDOM_STATE,
    preprocessing_steps: list[tuple[str, Any]] | None = None,
    n_jobs: int = 1,
) -> OptimizationResult:
    """Algoritmo Genético (DEAP), pipeline construido manualmente por individuo.

    Operadores (justificación)
    ---------------------------
    - **Selección — torneo** (``tournsize=tournament_size``): se prefiere
      sobre ruleta (fitness-proportionate) porque no requiere escalar/
      normalizar el fitness y controla la presión de selección de forma
      explícita vía ``tournament_size``, reduciendo el riesgo de que un
      único individuo domine tempranamente la población (causa común de
      convergencia prematura con ruleta).
    - **Crossover — uniforme** (``indpb=0.5``, aplicado con probabilidad
      ``cx_prob`` por par de padres): cada gen se intercambia
      independientemente; apropiado porque los hiperparámetros no tienen
      una noción de "posición adyacente" que un crossover de un punto
      debiera preservar.
    - **Mutación — gaussiana/categórica** (``indpb=mut_indpb`` por gen,
      aplicada con probabilidad ``mut_prob`` por individuo): ruido gaussiano
      (en escala log10 para hiperparámetros log) para genes numéricos,
      reasignación aleatoria para categóricos.
    - **Elitismo**: los ``elite_size`` mejores individuos de cada
      generación pasan intactos (sin cruce/mutación ni re-evaluación) a la
      siguiente, garantizando que el mejor score nunca empeore entre
      generaciones.

    Diversidad genética
    --------------------
    Se registra ``population_diversity()`` en cada generación
    (``diagnostics["diversity_history"]``) como indicador de convergencia
    prematura: si cae a ~0 mucho antes de agotar ``n_generations``, el GA
    dejó de explorar y el resultado depende fuertemente de la población
    inicial (sugiere aumentar ``population_size`` o ``mut_prob``).

    Parameters
    ----------
    population_size : int, default=config.GA_POPULATION_SIZE
    n_generations : int, default=config.GA_N_GENERATIONS
    cx_prob, mut_prob, mut_indpb : float
        Probabilidades de crossover/mutación (ver arriba).
    tournament_size : int, default=config.GA_TOURNAMENT_SIZE
    elite_size : int, default=config.GA_ELITE_SIZE
    n_jobs : int, default=1
        Individuos de la poblacion evaluados en paralelo (procesos, via
        joblib/loky) en vez de secuencialmente. Cada individuo se evalua de
        forma independiente (su propio nested cv.split() + pipeline.fit()),
        asi que paralelizar esta evaluacion no cambia que individuos se
        evaluan, cuantas generaciones corren, ni el resultado de cada
        evaluacion — solo el tiempo de pared. Con ``n_jobs=1`` (default) el
        comportamiento es identico al secuencial original (map() estandar).

    Returns
    -------
    OptimizationResult
    """
    random.seed(random_state)  # DEAP usa el módulo `random` global, no numpy

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    hyperparams = spec.hyperparameters
    scorer = get_scorer(scoring)
    n_splits = _n_splits(cv)

    def _init_individual():
        return creator.Individual([_init_gene(hp) for hp in hyperparams])

    def _decode(individual: list) -> dict[str, Any]:
        return {hp.name: gene for hp, gene in zip(hyperparams, individual)}

    def _evaluate(individual: list) -> tuple[float]:
        params = _decode(individual)
        fold_scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            pipe = build_pipeline(
                spec,
                technique,
                y_train=y_tr,
                preprocessing_steps=preprocessing_steps,
                random_state=random_state,
                **params,
            )
            pipe.fit(X_tr, y_tr)
            fold_scores.append(scorer(pipe, X_val, y_val))
        return (float(np.mean(fold_scores)),)

    toolbox = base.Toolbox()
    toolbox.register("individual", _init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", _cx_uniform, indpb=0.5)
    toolbox.register("mutate", _mutate_individual, hyperparams=hyperparams, indpb=mut_indpb)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("evaluate", _evaluate)

    def _map_evaluate(individuals: list) -> list[tuple[float]]:
        # n_jobs=1 preserva el map() secuencial original bit a bit; el
        # paralelo preserva el orden de salida, asi que el resultado es el
        # mismo con cualquier n_jobs, solo cambia el tiempo de pared (ver
        # docstring de n_jobs arriba).
        # backend="threading" (NO "loky"/procesos): con procesos, en la
        # corrida real, un worker terminaba con TerminatedWorkerError
        # (segfault) al evaluar individuos de svm -- diagnosticado de forma
        # aislada: el mismo individuo evaluado SECUENCIAL (sin
        # multiprocessing) termina bien, asi que el problema es especifico
        # de picklear/reconstruir el closure de _evaluate (captura X, y,
        # clases dinamicas de creator.Individual/FitnessMax) a traves de
        # procesos via loky, no del fit en si. Con threads no hay pickling
        # (memoria compartida) y ademas evita duplicar X/y por worker; SVC
        # libera el GIL durante el ajuste en C (libsvm), asi que threading
        # sigue dando paralelismo real. Ver EXPERIMENT_STATUS.md seccion 8n.
        if n_jobs == 1 or len(individuals) <= 1:
            return list(map(toolbox.evaluate, individuals))
        return Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(toolbox.evaluate)(ind) for ind in individuals
        )

    population = toolbox.population(n=population_size)
    for ind, fit in zip(population, _map_evaluate(population)):
        ind.fitness.values = fit
    n_evaluations = len(population)

    diversity_history = [population_diversity(population, hyperparams)]
    best_score_history = [max(ind.fitness.values[0] for ind in population)]

    for _generation in range(1, n_generations + 1):
        elite = [toolbox.clone(ind) for ind in tools.selBest(population, elite_size)]

        offspring = [toolbox.clone(ind) for ind in toolbox.select(population, len(population) - elite_size)]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, _map_evaluate(invalid)):
            ind.fitness.values = fit
        n_evaluations += len(invalid)

        population = elite + offspring
        diversity_history.append(population_diversity(population, hyperparams))
        best_score_history.append(max(ind.fitness.values[0] for ind in population))

    best_ind = tools.selBest(population, 1)[0]
    best_params = _decode(best_ind)
    best_pipeline = build_pipeline(
        spec,
        technique,
        y_train=y,
        cv_n_splits=n_splits,
        preprocessing_steps=preprocessing_steps,
        random_state=random_state,
        **best_params,
    )
    best_pipeline.fit(X, y)

    return OptimizationResult(
        method="genetic_deap",
        best_params=best_params,
        best_score=float(best_ind.fitness.values[0]),
        best_pipeline=best_pipeline,
        n_evaluations=n_evaluations,
        diagnostics={
            "selection": f"torneo (tournsize={tournament_size})",
            "crossover": f"uniforme (indpb=0.5, prob_por_par={cx_prob})",
            "mutation": f"gaussiana/categorica (indpb={mut_indpb}, prob_por_individuo={mut_prob})",
            "elitism": f"top-{elite_size} preservados sin cambios cada generacion",
            "population_size": population_size,
            "n_generations": n_generations,
            "diversity_history": diversity_history,
            "best_score_history": best_score_history,
        },
    )


# ---------------------------------------------------------------------------
# Dispatcher único — usado por nested_cv.py / experiment_runner.py
# ---------------------------------------------------------------------------

_DISPATCH: dict[str, Callable[..., OptimizationResult]] = {
    "grid_search": run_grid_search,
    "random_search": run_random_search,
    "bayesian_optuna": run_bayesian_search,
    "genetic_deap": run_genetic_search,
}


def run_optimization(
    method: str,
    spec: ModelSpec,
    technique: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    scoring: str = config.DEFAULT_SCORING,
    random_state: int = config.RANDOM_STATE,
    preprocessing_steps: list[tuple[str, Any]] | None = None,
    **kwargs: Any,
) -> OptimizationResult:
    """Punto de entrada único para cualquiera de los 4 métodos de optimización.

    Permite que ``nested_cv.py``/``experiment_runner.py`` iteren sobre
    ``optimization.OPTIMIZATION_METHODS`` con una sola llamada parametrizada,
    en vez de bifurcar manualmente por método (ver requisito de
    automatización del curso).

    Parameters
    ----------
    method : {"grid_search", "random_search", "bayesian_optuna", "genetic_deap"}
    spec, technique, X, y, cv, scoring, random_state
        Ver las funciones específicas de cada método.
    preprocessing_steps : list of (str, estimator), optional
        Ver ``balancing.build_pipeline`` (típicamente
        ``preprocessing.get_preprocessing_steps(spec)``).
    **kwargs
        Argumentos adicionales específicos del método (ej. ``n_iter`` para
        random_search, ``population_size``/``n_generations`` para
        genetic_deap).

    Returns
    -------
    OptimizationResult
    """
    if method not in _DISPATCH:
        raise ValueError(f"Método de optimización desconocido: '{method}'. Opciones: {OPTIMIZATION_METHODS}")
    return _DISPATCH[method](
        spec,
        technique,
        X,
        y,
        cv,
        scoring=scoring,
        random_state=random_state,
        preprocessing_steps=preprocessing_steps,
        **kwargs,
    )
