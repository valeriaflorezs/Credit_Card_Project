"""Configuración central del proyecto.

Toda ruta de datos, nombre de columna target y parámetro de validación
cruzada se define aquí — ningún otro módulo debe hardcodear estos valores.
"""

from pathlib import Path

# --- Reproducibilidad ---
RANDOM_STATE = 42

# --- Rutas ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "default_of_credit_card_clients.xls"  # dataset original (UCI), usado por el EDA
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "dataset_final.csv"  # lo genera la Parte 5 de notebooks/00_eda.ipynb (o `python -m src.preprocessing`)
RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENTS_MASTER_PATH = RESULTS_DIR / "experiments_master.csv"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
JOBLIB_CACHE_DIR = RESULTS_DIR / ".joblib_cache"  # cache de pasos de balanceo (ver balancing.py)

# --- Columna target ---
# Confirmado por el EDA de la compañera (2026-09-16): renombra
# "default payment next month" -> "default_payment_next_month". AJUSTAR
# solo si el dataset final entregado cambia este nombre.
TARGET_COLUMN = "default_payment_next_month"

# --- Columnas categóricas nominales ---
# Confirmado por el EDA de la compañera (2026-09-16): SEX/EDUCATION/MARRIAGE
# son categóricas nominales sin codificar (enteros; categorías inválidas ya
# reagrupadas en "Otros"). No hay valores nulos en el dataset — no se
# requiere imputación. AJUSTAR si el dataset final cambia nombres o
# agrega/quita columnas categóricas. Cualquier columna NO listada aquí se
# trata como numérica automáticamente (ver preprocessing.py).
CATEGORICAL_COLUMNS: list[str] = ["SEX", "EDUCATION", "MARRIAGE"]

# --- Validación cruzada anidada ---
N_OUTER_FOLDS = 3
N_INNER_FOLDS = 3

# --- Optimización de hiperparámetros ---
RANDOM_SEARCH_ITER = 25

# Bayesiana (Optuna) — sampler TPE (ver src/optimization.py para justificación)
OPTUNA_N_TRIALS = 30

# Genética (DEAP) — valores conservadores por defecto dado el hardware
# disponible (CPU-only, 16GB RAM); reducir aún más para modelos costosos
# (svm, knn) al invocar desde experiment_runner.py (bloque 6).
GA_POPULATION_SIZE = 16
GA_N_GENERATIONS = 10
GA_CX_PROB = 0.6          # prob. de aplicar crossover a un par de padres
GA_MUT_PROB = 0.3         # prob. de aplicar mutación a un individuo
GA_MUT_INDPB = 0.2        # prob. de mutar cada gen individual, dado que el individuo muta

# --- Paralelismo a nivel de metodo de optimizacion ---
# Maquina con 12 nucleos logicos. Se deja n_jobs=1 en el modelo (RandomForest/
# XGBoost, ver models.py) y en el bucle externo de experiment_runner.py (evita
# paralelismo anidado) — el paralelismo real se aplica UNA sola vez, en el eje
# de evaluaciones independientes de cada metodo (candidatos de grid/random
# search, trials de Optuna, individuos de la poblacion en genetic_deap). No
# cambia folds, rangos de hiperparametros, tamaño de poblacion/generaciones ni
# iteraciones — mismos resultados, menos tiempo de pared.
# Bajado de 12 a 8, y de 8 a 4 (2026-09-23, ~12:38): con 12 workers,
# svm/none/genetic_deap crasheo con TerminatedWorkerError (SO mato un worker
# por memoria) tras >4h; con 8 volvio a crashear en ~2 min. Diagnostico con
# 8: solo ~3GB de RAM genuinamente libres de los 16GB totales (Windows +
# OneDrive + el navegador y otras aplicaciones ya usan ~13GB en base) — mucho
# menos margen del que se asumio. svm con probability=True entrena 5
# modelos internos por fit (Platt scaling); con varios workers simultaneos
# sobre ~20K filas cada uno, el consumo de RAM supera rapido lo disponible.
# Con 2 (loky/procesos) TAMBIEN crasheo, en <1 min y con MAS RAM libre que
# antes -- eso descarto que fuera memoria. Diagnostico aislado (ver
# EXPERIMENT_STATUS.md seccion 8n): el mismo individuo evaluado secuencial
# (sin multiprocessing) termina bien -- el problema era especifico de
# picklear el closure de _evaluate a traves de procesos (loky), no del fit.
# Arreglado en optimization.py::run_genetic_search (backend="threading" en
# vez de "loky"). Con threads no hay ese riesgo de duplicar memoria por
# proceso (memoria compartida) -- subido de nuevo a 10 (2026-09-23, ~13:35,
# a pedido del usuario, aplicado para las combinaciones que faltan despues
# de que termine la que esta en curso, no afecta el proceso ya vivo).
OPTIMIZATION_N_JOBS = 10
GA_TOURNAMENT_SIZE = 3
GA_ELITE_SIZE = 2         # mejores individuos preservados sin cambios cada generación

# --- Métrica de validación por defecto ---
# Justificación completa en evaluation.py (bloque 8): dado el desbalance
# ~78/22 y el costo asimétrico de falsos negativos en riesgo crediticio,
# F1 pondera precision/recall sin ignorar la clase minoritaria como sí lo
# hace accuracy. Los optimizadores (bloque 4) y nested_cv (bloque 5) usan
# este valor por defecto, mientras evaluation.py reporta el set completo
# de métricas para la decisión final.
DEFAULT_SCORING = "f1"

# --- Datos sintéticos (mientras llega el dataset real) ---
SYNTHETIC_N_SAMPLES = 5000
SYNTHETIC_N_FEATURES = 20
SYNTHETIC_MAJORITY_RATIO = 0.78  # ~78/22, igual que el desbalance real esperado
