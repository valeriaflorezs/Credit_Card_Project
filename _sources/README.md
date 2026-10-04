# Default of Credit Card Clients — Pipeline de Clasificación (Entregable 3)

Pipeline de optimización de hiperparámetros y validación estadística para
112 modelos de clasificación (7 modelos base × 4 técnicas de balanceo ×
4 métodos de optimización) sobre el dataset "Default of Credit Card
Clients" (UCI, id=350), con validación cruzada anidada y comparación
estadística jerárquica (Friedman → Nemenyi → DeLong).

Ver [PROJECT_BRIEF.md](../../../Downloads/PROJECT_BRIEF.md) para el detalle completo de requisitos.

## Estado del pipeline (por bloque)

- [x] **Bloque 0** — Setup y reproducibilidad (`config.py`, `requirements.txt`, estructura de carpetas)
- [x] **Bloque 1** — Split + generador de datos sintéticos (`data_loader.py`, `synthetic_data.py`)
- [x] **Bloque 2** — Modelos base y espacios de búsqueda (`models.py`)
- [x] **Bloque 3** — Wrappers de balanceo (`balancing.py`)
- [x] **Bloque 4** — Métodos de optimización: Grid/Random/Bayesiana/Genética (`optimization.py`)
- [x] **Preprocesamiento** — ejecutado: dataset final en `data/processed/dataset_final.csv` (deduplicación, categorías reagrupadas, log de variables asimétricas, resumen de `BILL_AMT1-6`) + encoding y scaling condicional dentro del pipeline (`preprocessing.py`, ver Parte 5 de `00_eda.ipynb`)
- [x] **Bloque 5** — Validación cruzada anidada (`nested_cv.py`)
- [x] **Bloque 6** — Orquestador de experimentos (`experiment_runner.py`)
- [x] **Bloque 7** — Análisis de complejidad computacional (`complexity_analysis.py`)
- [x] **Bloque 8** — Evaluación y calibración (`evaluation.py`)
- [x] **Bloque 9** — Interpretabilidad SHAP/LIME (`interpretability.py`)
- [x] **Bloque 10** — Comparación estadística jerárquica (`stats_comparison.py`)

## Dataset y preprocesamiento

El dataset preprocesado ya está en `data/processed/dataset_final.csv`
(29,965 filas × 21 columnas, target `default_payment_next_month` al final).
Lo genera la Parte 5 de `notebooks/00_eda.ipynb`, que resuelve cada
hallazgo del EDA y verifica el resultado; también se puede regenerar desde
el archivo original con `python -m src.preprocessing`.

| Hallazgo del EDA | Resolución |
|---|---|
| 35 filas duplicadas | Eliminadas (antes de cualquier split) |
| Códigos no documentados en `EDUCATION` (0, 5, 6) y `MARRIAGE` (0) | Reagrupados en "Otros" |
| Saldos de facturación negativos | Conservados; resumidos en `N_CREDIT_BALANCE_MONTHS` |
| Asimetría extrema en `LIMIT_BAL` y `PAY_AMT1-6` | Log con signo (columnas `*_LOG`) |
| Multicolinealidad en `BILL_AMT1-6` (VIF hasta ~25) | Reemplazadas por `BILL_AMT_MEAN_6M_LOG` y `BILL_AMT_TREND_6M_LOG` (VIF máximo del dataset: 5.8) |
| Desbalance 78/22 | Dentro de la validación cruzada (`class_weight`, SMOTE, ADASYN) |
| Categóricas sin codificar / escalas distintas | One-hot y `StandardScaler` condicional dentro del pipeline |

Todo el código del pipeline es **agnóstico a los nombres de columnas
numéricas** — solo necesita el nombre de la columna target
(`src/config.py::TARGET_COLUMN`) y la lista de columnas categóricas
nominales (`src/config.py::CATEGORICAL_COLUMNS`), ya confirmados contra el
dataset real. `src.synthetic_data.generate_synthetic_dataset()` sigue
disponible para probar el pipeline sin datos reales (`03_pipeline_dev`).

**Resultados anteriores:** los resultados de `results/experiments_master.csv`
(y los notebooks `05`/`06` que los leen) se calcularon con la versión previa
del dataset (sin transformaciones log ni resumen de `BILL_AMT`). Esa versión
quedó respaldada en `data/processed/dataset_final_v1_antes_de_resolver_eda.csv`.
Para comparar modelos sobre el dataset actual hay que volver a correr `04`.

## Próximos pasos (modelado sobre el dataset preprocesado)

Los 11 bloques del pipeline están implementados y probados con datos
sintéticos y con el dataset real (ver docstrings de cada módulo para el
detalle de qué se verificó). Falta:

1. ~~Dataset final preprocesado~~ — hecho (ver sección anterior).
2. Revisar/mejorar los modelos y volver a correr `experiment_runner.run_experiments()` sobre las ~104
   combinaciones válidas reales (empezando por los modelos baratos,
   dejando SVM/KNN para el final o para Colab — ver
   `experiment_runner.MODEL_COST_ORDER`).
3. Con los resultados reales: `complexity_analysis.build_complexity_table`,
   `evaluation` (calibración/recalibración/sensibilidad a semillas),
   `interpretability` (SHAP/LIME del mejor modelo) y
   `stats_comparison.run_hierarchical_comparison` para la comparación
   jerárquica final.
4. Los 7 notebooks (`00`-`06`) ya están armados:
   - `00_eda`: EDA (dataset original UCI, en
     `data/raw/default_of_credit_card_clients.xls`) — análisis univariado,
     bivariado, multicolinealidad (VIF) e interpretabilidad SHAP sobre un
     Random Forest de referencia, y Parte 5 con el preprocesamiento que
     resuelve los hallazgos y guarda `data/processed/dataset_final.csv`.
     Los gráficos interactivos de las secciones 1.2 y 2.2 muestran, debajo,
     la interpretación de la variable elegida en el desplegable, con botones
     para alternar entre la versión básica y la técnica.
   - `01_baseline_logistic`: modelo base de Regresión Logística (hiperparámetros
     por defecto de scikit-learn, sin búsqueda todavía) sobre el dataset preprocesado —
     score (accuracy), error (log-loss), AUC-ROC, coeficientes, matriz de
     confusión y curva ROC. Deja documentado el margen de mejora frente al
     baseline ingenuo (predecir siempre "no default"), que el pipeline
     completo (`03`-`06`) busca cerrar con balanceo, optimización de
     hiperparámetros y comparación de 7 modelos.
   - `02_setup_check`: entorno, semillas, config, carga de datos sintéticos y del dataset real.
   - `03_pipeline_dev`: recorre los 11 bloques del pipeline en miniatura
     (checkpoint propio `results/dev_pipeline_check.csv`, gitignored —
     nunca toca `experiments_master.csv`); correr esto tras cualquier
     cambio de código para validar rápido que todo sigue funcionando junto.
   - `04_run_experiments`: corre el batch real (`QUICK_MODE` para iterar
     rápido; `False` para las ~104 combinaciones completas), guarda
     incremental/reanudable en `results/experiments_master.csv`.
   - `05_results_analysis` y `06_interpretability`: funcionan de forma
     independiente desde `results/experiments_master.csv` (no dependen de
     tener nada en memoria de un notebook anterior) — tabla de métricas,
     Friedman → Nemenyi → CD diagram → DeLong → Cliff's Delta, matriz de
     confusión/ROC, calibración (+ recalibración si corresponde),
     complejidad estándar vs. optimizado, y SHAP/LIME del mejor modelo.

## Cómo correr

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Luego abre `notebooks/00_eda.ipynb` para el análisis exploratorio sobre el
dataset original (ya incluido en `data/raw/`), `notebooks/01_baseline_logistic.ipynb`
para el modelo base de Regresión Logística, o
`notebooks/02_setup_check.ipynb` para verificar el entorno, semillas y la
carga de datos sintéticos.

## Presentación como Jupyter Book

Los notebooks están configurados para compilarse como un Jupyter Book
(`_config.yml` / `_toc.yml` en la raíz): cada celda de código queda
colapsada bajo un botón **"Show code cell source"** (se puede abrir si
hace falta revisar el código), dejando el foco en texto, tablas y
gráficos.

```bash
jupyter-book build .
```

Abre `_build/html/index.html` en el navegador. Si agregas o modificas
código, vuelve a ejecutar el notebook correspondiente (para refrescar sus
salidas) antes de recompilar — el book solo renderiza lo que ya está
guardado en cada `.ipynb` (`execute_notebooks: "off"` en `_config.yml`).

## Estructura

```
data/raw/                 # dataset original UCI (.xls), usado por 00_eda.ipynb
data/processed/           # dataset preprocesado final (no versionado)
src/                       # código del pipeline (ver docstrings de cada módulo)
notebooks/                 # desarrollo y ejecución interactiva
results/                   # tabla maestra de experimentos, figuras, modelos serializados
```
