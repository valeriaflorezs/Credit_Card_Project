# Estado de la corrida de experimentos (04_run_experiments.ipynb)

Última actualización: 2026-09-22, ~23:50 (autónomo, ver sección 8). Este archivo es el punto de partida para retomar la corrida en una sesión nueva.

> **Nota (2026-10-03): el dataset cambió.** Todo lo que sigue se corrió con la versión anterior de `data/processed/dataset_final.csv` (respaldada en `data/processed/dataset_final_v1_antes_de_resolver_eda.csv`: con las seis `BILL_AMT` por separado y sin transformaciones log; los duplicados se quitaban al cargar, en el notebook). La versión actual resuelve todos los hallazgos del EDA (ver Parte 5 de `notebooks/00_eda.ipynb` y el README): 29,965 filas × 21 columnas, `BILL_AMT1-6` resumidas en 3 variables, `LIMIT_BAL`/`PAY_AMT*` en log. Las 99 filas de `results/experiments_master.csv` son, por tanto, de otro dataset y no son comparables con una corrida nueva.

## 1. Progreso: 100/112 combinaciones guardadas — `svm/none/genetic_deap` terminó OK (2026-09-23 ~17:07) con el fix de threading, 15645.8s (~4h21min)

Archivo de resultados: `results/experiments_master.csv` (checkpoint incremental, una fila por combinación terminada).

| Modelo | Estado |
|---|---|
| `naive_bayes` | ✅ 16/16 completo |
| `logistic_regression` | ✅ 16/16 completo |
| `decision_tree` | ✅ 16/16 completo |
| `xgboost` | ✅ 16/16 completo |
| `random_forest` | ✅ 16/16 completo (terminado 2026-09-22 ~23:48, con paralelismo — ver sección 8) |
| `knn` | ✅ 16/16 completo (terminado 2026-09-23, muy rápido con paralelismo: genetic_deap 185-600s vs 86-157 min en random_forest) |
| `svm` | 🟡 5/16 (3 reales + 2 saltadas por timeout, documentadas) — pausado aquí, 11 pendientes (ver sección 8f) |

**Pausado al llegar al presupuesto de 7h** (autorizado por el usuario, "termina en las 7 horas"): 99/112 filas reales en `experiments_master.csv` + 2 combinaciones de svm saltadas por timeout externo y documentadas en `results/skipped_timeout_log.csv` (`svm/none/genetic_deap`, `svm/smote/grid_search`). Quedan 11 pendientes, TODAS de `svm` (ver sección 8f para el detalle y cómo continuar).

**Total teórico: 112** (7 modelos × 4 técnicas de balanceo × 4 métodos de optimización). Ya no hay combinaciones "no aplicables" — ver sección 3.

## 2. Cómo continuar

```bash
cd "C:/Users/valef/OneDrive/Desktop/Credit_Card_Project"
./.venv/Scripts/jupyter-nbconvert.exe --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "notebooks/04_run_experiments.ipynb" > logs/04_run_resumeN.log 2>&1
```

Correrlo en background. El notebook ya tiene `resume=True` en la llamada a `run_experiments()`, así que salta automáticamente las 75 combinaciones ya guardadas y sigue con `random_forest/adasyn/genetic_deap`.

**Antes de pausar**, siempre verificar que no queden procesos activos:
```bash
powershell -Command "Get-Process | Where-Object {\$_.ProcessName -match 'python|jupyter'}"
```

## 3. Ajustes hechos (autónomos, documentados también como comentarios en el código)

### a) Dataset real
- `data/processed/dataset_final.csv` construido desde `data/raw/default_of_credit_card_clients.xls` con las reglas ya confirmadas por el EDA (renombrar target, reagrupar EDUCATION/MARRIAGE en "Otros").
- **Se sacó la columna `ID`** — el primer intento la había dejado, lo que la habría convertido en una feature más (fuga/ruido). Corregido antes de correr nada.
- `notebooks/04_run_experiments.ipynb`: se quitó `QUICK_MODE`, ahora siempre corre la grilla completa con los presupuestos reales de `config.py`.

### b) Bug de explosión combinatoria en `grid_search` (src/models.py::get_param_grid)
- Con `n_points=4` fijo por hiperparámetro, modelos con muchos hiperparámetros continuos explotaban: XGBoost (6 hiperparámetros) daba 4⁶=4096 combinaciones × 9 folds ≈ 37,000 ajustes (~7+ horas, se colgó).
- **Fix**: se agregó `max_combinations` (tope general de 200) que reduce `n_points` automáticamente solo cuando hace falta. XGBoost bajó a 64 combinaciones.
- **Random Forest** seguía lento incluso con el tope de 200 (162 combinaciones, ~76+ min sin terminar una sola combinación — cada ajuste de Random Forest es más caro que uno de XGBoost porque scikit-learn no tiene la optimización de histogramas de `hist` en XGBoost). Se agregó un override por modelo (`ModelSpec.grid_search_max_combinations`), bajado a **80 para random_forest → 32 combinaciones reales**. Confirmado con el usuario antes de aplicarlo (pidió "reducir a 100"; con la fórmula actual solo hay dos valores limpios posibles, 162 o 32 — se explicó y se fue con 32).
- Esto **NO afecta** `random_search`/`bayesian_optuna`/`genetic_deap`: esos 3 métodos siguen explorando el espacio de hiperparámetros completo y sin recortar, para los 7 modelos. Solo `grid_search` (uno de los 4 métodos comparados) tiene el tamaño de grilla acotado, y solo para los modelos con muchos hiperparámetros continuos (xgboost, random_forest). Esto es metodológicamente esperable: es una limitación conocida de grid search exhaustivo, no un sesgo introducido.

### c) Cache de joblib para SMOTE/ADASYN (src/balancing.py)
- `_PIPELINE_CACHE = Memory(location=results/.joblib_cache)`, pasado a todo `ImbPipeline`.
- Hipótesis inicial: evitar recalcular SMOTE/ADASYN redundantemente en cada combinación de hiperparámetros de `grid_search`.
- **Resultado real (probado con datos reales)**: la mejora es modesta (~1.1x), NO dramática. La verdadera causa de la lentitud de `smote`/`adasyn` es que agrandan el dataset de entrenamiento (~1.5-1.6x más filas), no el recálculo redundante. Se dejó el cache igual porque es gratis y nunca cambia resultados (verificado: predicciones idénticas con/sin cache).

### d) 112 en vez de 104: `naive_bayes` y `knn` ahora sí soportan `class_weight`
KNN y Naive Bayes no tienen `class_weight` nativo en scikit-learn. Se investigó (notas del profesor + documentación de sklearn + búsqueda web) antes de decidir qué hacer:

- **`naive_bayes` → `BalancedGaussianNB`** (en `src/models.py`): `GaussianNB.fit()` sí acepta `sample_weight`. El wrapper traduce `class_weight='balanced'` a `sample_weight` vía `sklearn.utils.class_weight.compute_sample_weight` — **mecanismo estándar de sklearn**, no un hack.
- **`knn` → `BalancedKNeighborsClassifier`** (en `src/models.py`): scikit-learn NO tiene ningún mecanismo para esto (issue abierto sin resolver: [scikit-learn/scikit-learn#26062](https://github.com/scikit-learn/scikit-learn/issues/26062); tampoco existe ninguna librería de terceros mantenida — se buscó explícitamente). El wrapper implementa la idea en la etapa de predicción: el voto de cada vecino se pondera por la frecuencia inversa de su clase (además del peso por distancia si `weights='distance'`). **Es una extensión NO estándar**, documentada explícitamente como tal en el docstring de la clase y en las notas del `ModelSpec` de knn — el propio profesor tampoco hace esto en sus apuntes.
- Ambas clases son **idénticas a la original de sklearn cuando `class_weight=None`** (comportamiento default, sin regresión) — verificado con tests antes de correr nada real: `predict_proba` idéntico, compatibles con `sklearn.clone()` y `GridSearchCV`.
- `naive_bayes/class_weight` ya corrió completo (16/16 total para ese modelo). `knn/class_weight` todavía no ha corrido (knn no ha empezado).

### e) Archivo de resultados obsoleto apartado
- Existía un `experiments_master.csv` de una corrida de demostración vieja (`QUICK_MODE`, datos sintéticos). Se movió a `results/experiments_master_quickmode_demo_backup.csv` antes de empezar la corrida real, para que `resume=True` no se saltara combinaciones por error.

## 4. Lo que NO se tocó (y no se debe tocar sin autorización explícita)

- `N_OUTER_FOLDS`, `N_INNER_FOLDS`, `RANDOM_SEARCH_ITER`, `OPTUNA_N_TRIALS`, `GA_POPULATION_SIZE`, `GA_N_GENERATIONS` en `config.py` — parámetros metodológicos parejos para los 7 modelos.
- Los rangos (`low`/`high`) de los hiperparámetros de cada modelo en `models.py`.
- La estructura de validación cruzada anidada (`nested_cv.py`) — verificada contra las notas del profesor (sección "Validación cruzada anidada" de su curso), estructura idéntica: bucle externo para evaluar, bucle interno (solo sobre el fold de entrenamiento externo) para elegir hiperparámetros.
- Cualquier cosa que pudiera introducir fuga de datos (scaler/balanceo siempre dentro del `Pipeline`, ajustado solo en fit()).

## 5. Regla para decidir si intervenir cuando una combinación tarda mucho

1. ¿Es por el tamaño de la grilla de `grid_search` (muchos hiperparámetros continuos)? → Sí se puede ajustar `grid_search_max_combinations` por modelo (ver 3b), documentando el cambio.
2. ¿Es por el costo inherente del algoritmo o por cómo el GA/random search explora el espacio (ej. `genetic_deap` tiende a converger hacia configuraciones más caras)? → **No tocar nada**, es esperado y metodológicamente correcto. Ejemplos ya observados: `random_forest/none/genetic_deap` tardó 157.6 min, `xgboost/*/grid_search` con `smote` tardó hasta 66.7 min — ninguno de los dos es un bug, es SVM/Random Forest/XGBoost siendo inherentemente costosos combinados con datasets agrandados por oversampling.
3. Nunca reducir `RANDOM_SEARCH_ITER`/`OPTUNA_N_TRIALS`/`GA_POPULATION_SIZE`/`GA_N_GENERATIONS` para "arreglar" la lentitud — rompe la comparación justa entre los 7 modelos.

## 6. Alerta para lo que sigue: `svm`

Todavía no ha corrido ninguna combinación de `svm`. El propio código del proyecto ya lo documenta como "el modelo más costoso del pipeline" (kernel RBF con complejidad O(n²)-O(n³) sobre ~20-30K filas, más `probability=True` que agrega una validación cruzada interna de 5 folds en libsvm). Es razonable esperar que sea el modelo más lento de los 7. Si alguna combinación de `svm` se comporta de forma anómala (no solo lenta, sino sin avanzar en CPU durante mucho tiempo), ahí sí vale la pena investigar más a fondo (podría ser un problema real, no solo lentitud esperada).

## 8. Autónomo (2026-09-22, ~22:15 en adelante): paralelismo + tope de tiempo

Instrucción del usuario: presupuesto de 7h para terminar las 112, con permiso de "pasar al siguiente" si algo se dificulta, pero **priorizando primero acelerar de verdad la implementación sin comprometer la metodología**, antes que simplemente recortar y saltar. A partir de este punto se trabajó de forma autónoma; se documenta cada decisión aquí.

### a) Paralelismo real (nuevo) — sin tocar folds/rangos/tamaños

Diagnóstico: `RandomForestClassifier`/`XGBClassifier` ya tenían `n_jobs=1` "a propósito" (comentario original: el paralelismo se manejaría a nivel de `experiment_runner.py`), pero el bucle externo de `experiment_runner.run_experiments()` también corre con `n_jobs=1` (secuencial) — es decir, **nunca hubo paralelismo real en ningún nivel**, con 12 núcleos lógicos disponibles sin usar.

Cambio: se paraleliza el eje de evaluaciones INDEPENDIENTES dentro de cada método de optimización (no el modelo, no el bucle de combinaciones — evita anidar paralelismo):
- `grid_search`/`random_search`: ya aceptaban `n_jobs` (GridSearchCV/RandomizedSearchCV nativos de sklearn) — solo se subió el valor.
- `bayesian_optuna` (`src/optimization.py::run_bayesian_search`): se agregó parámetro `n_jobs`, pasado a `study.optimize(n_jobs=...)` (soporte nativo de Optuna, hilos). Con `n_jobs>1` puede haber varios trials "en vuelo" evaluados contra el mismo estado previo del study en vez de estrictamente secuencial — comportamiento documentado de Optuna, no un cambio metodológico.
- `genetic_deap` (`src/optimization.py::run_genetic_search`): se agregó parámetro `n_jobs`; la evaluación de la población (inicial y de cada generación) se paraleliza con `joblib.Parallel(backend="loky")` en vez de `map()` secuencial. **Verificado**: con `n_jobs=1` el comportamiento es idéntico al original; con `n_jobs>1` el resultado (`best_score`/`best_params`) es **bit a bit idéntico** al secuencial en la prueba aislada (`joblib.Parallel` preserva el orden de salida) — mismos individuos evaluados, mismas generaciones, solo cambia el tiempo de pared.
- `config.OPTIMIZATION_N_JOBS = 10` (nuevo): 12 núcleos lógicos disponibles, se dejan 2 libres para el SO/monitoreo. Se usa el mismo valor para los 4 métodos vía `optimizer_kwargs_by_method` en `04_run_experiments.ipynb` (bloque de configuración).
- El paralelismo del modelo (`RandomForestClassifier`/`XGBClassifier` `n_jobs=1`) y del bucle externo (`experiment_runner` `n_jobs=1`) **no se tocó** — un solo nivel de paralelismo activo a la vez, para no sobre-suscribir los 12 núcleos.

Esto es una optimización de **implementación**, no de metodología: mismos folds, mismos rangos de hiperparámetros, mismo `GA_POPULATION_SIZE`/`GA_N_GENERATIONS`/`RANDOM_SEARCH_ITER`/`OPTUNA_N_TRIALS` para los 7 modelos — solo se usa hardware que estaba ocioso.

### b) Tope de tiempo por combinación (backstop, no la primera línea de defensa)

`src/experiment_runner.py::run_experiments(timeout_seconds=...)` (nuevo, default `None` = comportamiento original sin cambios): cada combinación corre en un `multiprocessing.Process` aparte; si excede `timeout_seconds` se cancela (`terminate()`), se registra en `results/skipped_timeout_log.csv` (NUNCA en `experiments_master.csv` — la tabla maestra solo tiene corridas con métricas reales) con motivo y tiempo transcurrido, y se sigue con la siguiente. `resume=True` excluye tanto lo ya completado como lo ya saltado por timeout (no se reintenta solo, hay que acotar explícitamente `models`/`techniques`/`methods` a esas combinaciones para reintentarlas).

Valor elegido: **45 minutos** (confirmado con el usuario). Pendiente al final de la corrida: **si sobra tiempo del presupuesto de 7h, reintentar las combinaciones de `skipped_timeout_log.csv`** (autorizado explícitamente por el usuario) — probablemente ya no hagan falta gracias al paralelismo de (a), pero queda como red de seguridad para casos genuinamente atípicos (ej. SVM con probability=True, que ya se documentó como el más costoso).

Probado de forma aislada antes de tocar la corrida real (deadline propio, tiempo transcurrido corriendo en paralelo sin afectar el proceso en vivo de random_forest): timeout forzado (se cancela y registra correctamente, no toca `experiments_master.csv`), combinación real completa dentro del tope (se registra normal), y `resume=True` no reintenta ni lo completado ni lo saltado.

### c) Corrección: el tope de tiempo (b) se colgó — desactivado

Al aplicar (a)+(b) juntos en la corrida real, `random_forest/class_weight/*` se quedó colgado (~35 min sin avanzar, CPU de los workers totalmente plano — no lento, literalmente sin cambiar ni un segundo de CPU en 15s de muestreo repetido). Causa: el wrapper de `timeout_seconds` mete cada combinación en su propio `multiprocessing.Process`, y ESE proceso a su vez llama a un método con `n_jobs=10` (que también usa `multiprocessing`/loky para paralelizar) — paralelismo de procesos anidado dentro de otro proceso (spawn dentro de spawn en Windows), que se colgó en vez de simplemente ser ineficiente.

**Corrección**: `timeout_seconds=None` en `04_run_experiments.ipynb` (desactivado). El paralelismo de (a) queda activo (es lo que de verdad importa para el presupuesto de 7h) sin el wrapper de proceso extra. El backstop para una combinación genuinamente atípica se maneja por fuera, con un monitor externo que compara el CPU total de los workers entre chequeos (~30s) y avisa si se mantiene plano ~3 minutos seguidos — sin matar nada automáticamente, para poder decidir caso por caso en vez de arriesgar otro falso positivo.

El código de `run_experiments(timeout_seconds=...)` en `src/experiment_runner.py` queda en el repo (probado de forma aislada y correcto en sí mismo — el problema es específicamente la combinación con `n_jobs>1` a nivel de método, no el mecanismo en sí), documentado para no reusarlo junto con paralelismo a nivel de método sin resolver antes el anidamiento (ej. usar backend `"threading"` en vez de `"loky"` para el wrapper, o mover el tope de tiempo a nivel de método en vez de por combinación completa).

### e) SVM: tope externo diferenciado por método (2026-09-23, ~01:30)

`svm/none/grid_search` (la grilla más chica de las 4 de svm: solo 16 combinaciones, `C`×`gamma`) terminó en 3399.6s (~57 min) — mucho más lento que cualquier otro modelo incluso con paralelismo, consistente con lo ya documentado (RBF O(n²)-O(n³) + `probability=True` agrega un CV interno de 5 folds por cada fit). Con ~3h48min restantes del presupuesto de 7h y 15 combinaciones de svm pendientes, seguir sin tope habría arriesgado todo el presupuesto en una sola combinación.

Proyección (evaluaciones, no tiempo — mismo `n_jobs`, así que el tiempo escala aprox. igual): `grid_search` evalúa 16 combinaciones × 3 folds internos = 48; `genetic_deap` evalúa `GA_POPULATION_SIZE + (GA_POPULATION_SIZE - GA_ELITE_SIZE) × GA_N_GENERATIONS` ≈ 16 + 14×10 = 156 individuos × 3 folds internos ≈ **~10x más evaluaciones que grid_search**. Con svm ya en ~57 min para el método más barato, `genetic_deap` corriendo al mismo ritmo tomaría horas — inviable dentro del presupuesto.

Como el wrapper interno (`timeout_seconds` en `run_experiments`) se cuelga anidado con `n_jobs>1` (sección 8c), se implementó un vigilante EXTERNO (`watch_and_cap.py`, corre en un proceso Python aparte, fuera del notebook): identifica la combinación pendiente actual (primera en el orden — determinista dado que la corrida es secuencial), le asigna un tope según el método, y espera a que aparezca una fila nueva en `experiments_master.csv` o se cumpla el tope. Si se cumple el tope, registra la combinación en `results/skipped_timeout_log.csv` (mismo esquema que el mecanismo interno) — así que aunque el mecanismo cambió, el rastro documentado es consistente — y yo (agente) mato los procesos viejos y relanzo el notebook, que la salta automáticamente vía `resume=True`.

Topes elegidos:
- `grid_search`/`random_search`/`bayesian_optuna`: **90 min** (generoso — ya vimos que grid_search necesita genuinamente ~57 min; se le da margen para terminar en vez de forzar un corte artificial).
- `genetic_deap`: **30 min** (bajo, a propósito — dada la proyección de ~10x evaluaciones, es muy poco probable que termine en ese tiempo, pero se le da una oportunidad real acotada en vez de saltarlo sin intentarlo, honrando la instrucción de "priorizar acelerar, no solo saltar").

Con estos topes, el peor caso (los 15 restantes agotan su tope) sigue excediendo el presupuesto (~11×90min + 4×30min ≈ 18.8h) — es decir, **es matemáticamente esperable que no se complete el 112/112 dentro de las 7h**, incluso con el paralelismo. Se prioriza cobertura amplia (grid/random/bayesian para las 4 técnicas restantes) sobre profundidad (genetic_deap, el más caro) — si sobra tiempo al final, se reintentan las saltadas (instrucción explícita del usuario).

### f) Estado al cumplirse el presupuesto de 7h (2026-09-23, ~05:41)

Con el ciclo vigilar→(éxito o timeout)→matar→relanzar de (e) funcionando, se completaron 3 combinaciones reales más de svm (`none/random_search` 3822.4s, `none/bayesian_optuna` 3902.1s — ambas ~65 min, dentro del tope de 90) y se saltaron 2 por exceder el tope (`none/genetic_deap` a los 1801.4s del tope de 1800s — justo como se proyectó; `smote/grid_search` a los 5405.4s del tope de 5400s — SMOTE aumenta el dataset de entrenamiento ~1.5-1.6x filas, lo que sobre un modelo O(n²)-O(n³) como SVM explica que ni siquiera grid_search, el método más barato, entrara en 90 min con esta técnica).

**Se llegó al límite de 7h con 99/112 filas reales + 2 saltadas documentadas (101/112 resueltas). Quedan 11 combinaciones pendientes, todas de svm**: `smote/{random_search,bayesian_optuna,genetic_deap}`, `adasyn/{grid_search,random_search,bayesian_optuna,genetic_deap}`, `class_weight/{grid_search,random_search,bayesian_optuna,genetic_deap}`.

**Para continuar** (fuera del presupuesto de 7h, requiere autorización explícita del usuario para seguir):
```bash
cd "C:/Users/valef/OneDrive/Desktop/Credit_Card_Project"
./.venv/Scripts/jupyter-nbconvert.exe --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "notebooks/04_run_experiments.ipynb" > logs/04_run_resumeN.log 2>&1
```
`resume=True` sigue saltando automáticamente las 99 ya completas Y las 2 ya registradas en `results/skipped_timeout_log.csv` (no las reintenta solas). Dado el patrón observado (smote ya mostró ser más lento que none; adasyn probablemente similar a smote ya que también sobremuestrea; class_weight no agranda el dataset así que debería comportarse más como `none`), lo esperable es:
- `class_weight/*`: probablemente terminen dentro de topes similares a `none` (~57-65 min grid/random/bayesian, `genetic_deap` seguramente salte igual que en `none`).
- `smote/*` y `adasyn/*` restantes: alto riesgo de exceder 90 min incluso en grid/random/bayesian, dado que `smote/grid_search` ya lo hizo.

Si se retoma, considerar subir el tope de grid/random/bayesian a ~120 min específicamente para smote/adasyn (documentando el cambio), o aceptar más saltos — el usuario ya autorizó reintentar las saltadas al final si sobra tiempo ("si pasas todos, puedes reintentar los pendientes").

### r) Pausado por el usuario (2026-09-24, ~06:5x)

Detenido limpio a pedido del usuario. **Estado: 104/112.** Completadas desde el reordenamiento (q): `class_weight/grid_search`, `class_weight/random_search`, `class_weight/bayesian_optuna`, `smote/grid_search`. Quedan 8 pendientes: `smote/random_search`, `smote/bayesian_optuna`, `smote/genetic_deap`, `adasyn/*` (las 4), `class_weight/genetic_deap`. El usuario está explorando en paralelo la vía de GPU en Colab (notebook `07_svm_gpu_colab.ipynb`, que falló por un problema conocido de `rapidsai-csp-utils` — requiere reiniciar el runtime de Colab después de instalar RAPIDS antes de importar `cuml`; se armó un prompt para que otro agente lo arregle). Para retomar local: mismo comando de siempre (sección 2), la configuración reordenada (`techniques=["class_weight","smote","adasyn"]`, `methods` sin `genetic_deap`) sigue activa en el notebook.

### q) Retomado con presupuesto de 6h — reordenado para maximizar cobertura (2026-09-23, ~17:20)

El usuario autorizó continuar con un nuevo presupuesto de 6h y "cualquier estrategia" para aprovechar el tiempo. Con datos reales ya medidos (grid_search/random_search/bayesian_optuna en `none`: 57/64/65 min; `genetic_deap` en `none`: 4h21min), la matemática no cierra para las 12 combinaciones restantes en 6h bajo el orden por defecto (que intercala `genetic_deap` cada 4 combinaciones) — ni siquiera las 9 combinaciones "baratas" solas caben cómodas (9 × ~60-90min ≈ 9-13.5h).

**Estrategia**: en vez de un tope de tiempo artificial (que en la fase anterior generó falsos positivos y complejidad), se **reordena** la corrida para maximizar combinaciones REALES completadas: `04_run_experiments.ipynb` ahora pasa `techniques=["class_weight", "smote", "adasyn"]` (class_weight primero, más barato — no agranda el dataset como smote/adasyn) y `methods=["grid_search", "random_search", "bayesian_optuna"]` (excluye `genetic_deap` de esta primera pasada). `genetic_deap` (el más caro, ~4h+ por técnica) se corre aparte después, solo si sobra presupuesto, para no arriesgar consumir las 6h enteras en un solo intento sin dejar nada más completo. `resume=True` sigue funcionando igual — no se pierde nada de lo ya hecho.

### p) Pausado por el usuario (2026-09-23, ~17:10)

`svm/smote/grid_search` (primera de las 12 restantes) acababa de arrancar bajo `04_run_resume17.log` (n_jobs=10, cache_size=500) cuando el usuario pidió pausar. Detenido limpio, sin pérdida más allá de lo ya sabido (nada se había guardado aún de esa combinación). **Estado: 100/112**, 12 pendientes (todas `smote`/`adasyn`/`class_weight` × 4 métodos). Para retomar: mismo comando de siempre (sección 2) — el notebook ya tiene los ajustes de (o) guardados en `config.py`/`models.py`, no hace falta tocar nada.

### o) `svm/none/genetic_deap` completó con el fix — relanzado con n_jobs=10 (2026-09-23, ~17:07)

**100/112.** El fix de threading funcionó de punta a punta en la corrida real completa (no solo en la prueba mini): 15645.8s (~4h21min), sin crashes, sin errores. Relanzado como `04_run_resume17.log` con los ajustes acordados con el usuario (`OPTIMIZATION_N_JOBS=10`, `cache_size=500` en SVC) para las 12 combinaciones restantes (`smote`/`adasyn`/`class_weight` × 4 métodos cada una).

### n) Causa raíz real encontrada: no era memoria, era `loky` (procesos) — arreglado con `threading` (2026-09-23, ~12:45)

Con `OPTIMIZATION_N_JOBS=2` volvió a crashear (mismo `TerminatedWorkerError`) en **menos de 1 minuto**, con MÁS RAM libre que en intentos anteriores (3.43GB) — esto descartó la memoria como causa. Diagnóstico aislado (`diagnose_svm_crash.py`, en el scratchpad): se evaluó la población inicial de `svm/none/genetic_deap` (misma `random_state`, así que son exactamente los mismos individuos de todos los intentos anteriores) **secuencialmente, sin multiprocessing**. El individuo `[0]` (`C=3.61, gamma=0.000126`) terminó bien en 258s — sin crash.

Conclusión: el problema no era el fit de SVM en sí, era específico de `loky` (backend de procesos de joblib): el closure `_evaluate` dentro de `run_genetic_search` captura `X`, `y` y las clases dinámicas de `creator.Individual`/`creator.FitnessMax` (DEAP) — picklearlas/reconstruirlas a través de procesos (spawn en Windows) resultó frágil.

**Fix**: `src/optimization.py::run_genetic_search` — `_map_evaluate` cambia de `backend="loky"` a `backend="threading"`. Sin pickling (memoria compartida entre hilos) se elimina la clase de error por completo, y como `SVC.fit()` libera el GIL durante el cómputo en C (libsvm), sigue habiendo paralelismo real. **Verificado con una corrida mini real** (svm, población=4, 1 generación, `n_jobs=2`, dataset real): `OK sin crash: best_score=0.4541, elapsed=1284.2s, n_evaluations=6` — además confirmó que el `max_iter=200_000` de (j) también está funcionando (`ConvergenceWarning: Solver terminated early`, uno de los 6 fits lo alcanzó).

Con esto arreglado, `OPTIMIZATION_N_JOBS` se sube de nuevo a **8** (ya no hay riesgo de duplicación de memoria por proceso). Relanzado como `04_run_resume16.log`.

### m) 4 workers también crasheó — bajado a 2, más limpieza de huérfanos (2026-09-23, ~12:40)

Con `OPTIMIZATION_N_JOBS=4` volvió a fallar en ~1-2 min (mismo `TerminatedWorkerError`). Se encontraron procesos `python`/`python3.13` huérfanos de la corrida anterior (no se limpiaron solos al fallar el notebook) — se mataron, pero la RAM libre NO mejoró significativamente (2.54GB vs 2.77GB antes), confirmando que el problema no es acumulación de huérfanos sino que la máquina simplemente tiene poco margen disponible en este momento (Windows + OneDrive + el navegador y otras aplicaciones ya usan la mayor parte de los 16GB, fuera del control de este proceso). Bajado a **`OPTIMIZATION_N_JOBS=2`**. Relanzado como `04_run_resume15.log`. **Rutina agregada**: verificar y matar procesos `python`/`jupyter` huérfanos antes de cada relanzamiento, no solo cuando se detecta un cuelgue.

### l) 8 workers tampoco alcanzó — la RAM libre real es ~3GB, no 16GB (2026-09-23, ~12:38)

Con `OPTIMIZATION_N_JOBS=8`, el mismo `TerminatedWorkerError` volvió a aparecer, esta vez en ~2 minutos (no horas) — señal de que no era una acumulación lenta, sino falta de margen desde el arranque. Verificado: de los 16GB totales de la máquina, solo ~3GB estaban genuinamente libres (Windows + OneDrive + el navegador y otras aplicaciones ya usan ~13GB en base, independiente de los experimentos). Bajado a **`OPTIMIZATION_N_JOBS=4`**. Relanzado como `04_run_resume14.log`.

### k) La causa real no era lentitud, era un crash por memoria (2026-09-23, ~12:35)

Antes de que se cumpliera la hora de margen de (j), el proceso terminó solo con error: `TerminatedWorkerError` — un worker fue matado por el sistema operativo (uso excesivo de memoria o segfault) tras >4h corriendo. Diagnóstico: `svm` con `probability=True` entrena 5 modelos internos por cada fit (Platt scaling) sobre ~20K filas; con `OPTIMIZATION_N_JOBS=12` (subido de 10 a 12 en la sección 8i a pedido del usuario) había 12 de estos fits pesados corriendo simultáneamente, saturando los 16GB de RAM de la máquina. El cuello de botella real de `svm` no es CPU — es memoria — así que más núcleos en paralelo no ayuda tanto como en otros modelos y aumenta el riesgo de OOM.

**Corrección**: `OPTIMIZATION_N_JOBS` bajado de 12 a **8** (más RAM disponible por worker). Junto con el `max_iter=200_000` de (j) (protege contra no-convergencia genuina) y el nuevo tope de memoria implícito (menos procesos simultáneos), las dos causas conocidas de "se cuelga para siempre" están mitigadas. Relanzado como `04_run_resume13.log` — se reintenta `svm/none/genetic_deap` desde cero (sin guardado parcial, como ya se explicó). Dado que ambas causas raíz están corregidas, se decide NO reimponer el límite artificial de 1h — se vuelve a la vigilancia por cuelgue real (CPU plano) únicamente.

### j) `svm/none/genetic_deap` colgado en un individuo patológico (2026-09-23, ~12:15)

A las 4h de corrida (arrancó 8:11am, sin resultado a las 12:12pm), diagnóstico: 16 workers activos pero con distribución de CPU muy despareja (un proceso con 8283s acumulados, más del doble que el segundo; dos workers casi sin uso) — consistente con que la mayoría de la población ya se evaluó y UN individuo (una combinación C/gamma específica) está tardando desproporcionadamente en converger en el solver SMO de libsvm. No es un cuelgue (ese proceso sigue sumando CPU), pero sí un caso atípico.

Se descubrió que `_svm_factory` (src/models.py) **no tenía `max_iter` acotado** (default de sklearn: `-1`, sin límite) — en el peor caso, sin garantía de terminar nunca. Se agregó `max_iter=200_000` (generoso, no afecta convergencia normal; si se alcanza el tope sin converger, sklearn usa la mejor solución parcial y esa combinación simplemente puntúa peor en el CV — comportamiento esperado, no invalida la búsqueda). **Este cambio NO afecta la corrida en curso** (código ya cargado en memoria del proceso viejo), solo protege reintentos futuros.

Como no hay guardado parcial por fold outer (nested_cv solo escribe la fila al terminar los 3 folds), cortar esta combinación pierde el 100% de las 4h invertidas — no hay forma de "recuperar" ese progreso. Discutido con el usuario: se decidió dar **1 hora más de margen firme** (no indefinido) antes de cortar y seguir con las 12 combinaciones restantes de svm.

### i) Retomado a máximo rendimiento (2026-09-23, ~07:30)

El usuario pidió seguir, usando el máximo del PC (queda conectado a la corriente, sin supervisión). `config.OPTIMIZATION_N_JOBS` subido de 10 a **12** (todos los núcleos lógicos — ya no se reservan 2 para el SO/monitoreo). Se activó `request_keep_awake` para evitar que el equipo entre en reposo por inactividad durante corridas largas sin interacción (nota: esta protección se libera sola tras ~5 min sin actividad de turno; dado que este PC va a quedar conectado a la corriente, sus propios ajustes de energía son la salvaguarda principal — igual conviene confirmar que Windows no esté configurado para dormir con el cargador conectado). Relanzado como `04_run_resume12.log`.

### h) Detenido por el usuario (2026-09-23, ~07:25)

`svm/none/genetic_deap` estaba en curso (reintentándose, ~66 min invertidos, CPU sano, sin errores) cuando el usuario pidió detener todo. Se mató el proceso (checkpoint intacto: nada se pierde salvo el trabajo en curso de esa combinación, que no se había guardado — `resume=True` la reintentará desde cero la próxima vez). **Estado al detener: 99/112 filas reales**, 13 pendientes (todas de `svm`, incluyendo las 2 que se habían saltado por timeout antes de que se autorizara seguir sin presupuesto — ver sección 8g/8f). `results/skipped_timeout_log.csv` vacío/no existe (se limpió en 8g). El archivo de las 2 saltadas originalmente sigue en `results/skipped_timeout_log_archive_before_full_completion.csv` por si hace falta consultarlo.

Para retomar: mismo comando de siempre (sección 2), `resume=True` sigue el orden por costo y no repite las 99 ya completas.

### g) El usuario autorizó continuar más allá de las 7h hasta 112/112

Con eso, se removió el tope artificial de tiempo: ahora el vigilante externo (`watch_and_cap.py`) deja correr cada combinación hasta que termine, y solo interviene si detecta un CUELGUE REAL (CPU total de los workers exactamente plano por ~12 minutos seguidos) — en ese caso, la acción es **reintentar la misma combinación** (matar y relanzar, no marcarla como saltada), porque ya no hay presión de presupuesto que justifique sacrificar cobertura. Las 2 combinaciones previamente saltadas por timeout (`svm/none/genetic_deap`, `svm/smote/grid_search`) se archivaron en `results/skipped_timeout_log_archive_before_full_completion.csv` y se limpiaron de `results/skipped_timeout_log.csv` para que se reintenten con las demás.

### d) Combinación en curso al momento de este cambio

`random_forest/adasyn/genetic_deap` seguía corriendo bajo el código VIEJO (sin paralelismo, sin tope) cuando se hizo este cambio — el proceso en memoria no puede recargar código nuevo. Decisión explícita del usuario: dejarla terminar sin límite (ya llevaba >1h20min invertidas), y aplicar el paralelismo/tope solo de ahí en adelante. Un monitor automático espera a que esa fila aparezca en `experiments_master.csv` (12/16 de random_forest) o a que el proceso muera/falle, y en ese momento se mata el proceso viejo (que de todas formas ya seguiría con el código antiguo para las combinaciones siguientes) y se relanza `04_run_experiments.ipynb` con los cambios de (a)+(b) ya cargados.

## 7. Archivos modificados en esta sesión

- `src/config.py` — `JOBLIB_CACHE_DIR` agregado.
- `src/models.py` — `BalancedGaussianNB`, `BalancedKNeighborsClassifier`, `get_param_grid` con `max_combinations`, `ModelSpec.grid_search_max_combinations`, registros de `knn`/`naive_bayes`/`random_forest` actualizados.
- `src/balancing.py` — cache de joblib en `build_pipeline`.
- `src/optimization.py` — `run_grid_search` usa `spec.grid_search_max_combinations`.
- `notebooks/04_run_experiments.ipynb` — sin `QUICK_MODE`, corre siempre la grilla completa.
- `notebooks/05_results_analysis.ipynb` — celda nueva que arma la tabla de 112 combinaciones (ya no debería haber ninguna "No aplica": se actualiza sola porque lee `is_compatible()` en vivo).
- `data/processed/dataset_final.csv` — nuevo, dataset real sin columna `ID`.
- `src/config.py` — `OPTIMIZATION_N_JOBS = 10` agregado (ver sección 8a).
- `src/optimization.py` — `n_jobs` en `run_bayesian_search` (Optuna) y `run_genetic_search` (evaluación de población paralela vía joblib/loky); ver sección 8a.
- `src/experiment_runner.py` — `run_experiments(timeout_seconds=..., skip_log_path=...)` nuevo (backstop, ver sección 8b); `results/skipped_timeout_log.csv` nuevo (no versionado en `experiments_master.csv`).
- `notebooks/04_run_experiments.ipynb` — bloque de configuración actualizado con `n_jobs` por método; `timeout_seconds` quedó en `None` (ver sección 8c, se probó pero se desactivó por colgarse anidado con el paralelismo).
