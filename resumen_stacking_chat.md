# Stacking para CS-FOF: resumen de la conversación

**Proyecto:** *Bosque Oblicuo de Fisher Sensible al Costo para la Predicción del Incumplimiento de Pago en
Tarjetas de Crédito* (repositorio `valeriaflorezs/Credit_Card_Project`).
**Fecha:** noviembre de 2026.
**Rama con el código nuevo:** `claude/festive-clarke-dm98im`.

---

## 1. ¿Qué es el stacking en machine learning?

El **stacking** (*stacked generalization*) es una técnica de ensamble que **combina las predicciones de
varios modelos distintos mediante otro modelo, el meta-modelo, que aprende a mezclarlas**.

**Cómo funciona**

1. **Nivel 0, modelos base:** se entrenan varios modelos distintos con los mismos datos (por ejemplo,
   regresión logística, Random Forest, XGBoost y KNN).
2. **Nivel 1, meta-modelo:** las **predicciones** de los modelos base se usan como **nuevas variables**,
   y con ellas se entrena un modelo final, normalmente sencillo, como una regresión logística.

```
Datos ──► Modelo A ──► pred_A ─┐
      ──► Modelo B ──► pred_B ─┼──► Meta-modelo ──► Predicción final
      ──► Modelo C ──► pred_C ─┘
```

**Detalle clave: predicciones *out-of-fold*.** Si el meta-modelo aprende con predicciones que los modelos
base hicieron sobre los mismos datos con los que se entrenaron, hay sobreajuste. Por eso cada modelo base
predice cada pliegue después de entrenarse con los demás, y el meta-modelo solo ve predicciones sobre
datos "no vistos".

**Por qué funciona:** cada modelo se equivoca en sitios distintos. El meta-modelo aprende en cuál confiar
y cuándo.

| Técnica | Cómo combina |
|---|---|
| Bagging (Random Forest) | Promedia modelos del mismo tipo entrenados con muestras distintas |
| Boosting (XGBoost) | Entrena modelos en secuencia; cada uno corrige al anterior |
| Stacking | Combina modelos **distintos** mediante un modelo que **aprende** la combinación |

**Ventajas:** suele mejorar la precisión y aprovecha modelos distintos.
**Desventajas:** es más costoso, menos interpretable (importa en riesgo crediticio) y mejora poco si los
modelos base son muy parecidos.

```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

stack = StackingClassifier(
    estimators=[("rf", RandomForestClassifier()), ("xgb", XGBClassifier())],
    final_estimator=LogisticRegression(),
    cv=5,                         # predicciones out-of-fold
    stack_method="predict_proba",
)
```

---

## 2. ¿Le sirve el stacking a nuestro modelo original (CS-FOF)?

**Respuesta corta:** probablemente da una mejora pequeña (milésimas de F1) y difícil de probar como
significativa, pero vale la pena intentarlo.

**A favor**

- Los mejores modelos están empatados en la validación anidada 5 × 5: CatBoost 0.5480, CS-FOF 0.5479,
  XGBoost 0.5466, LightGBM 0.5462 y Random Forest 0.5452.
- CS-FOF y XGBoost se equivocan en sitios distintos:
  - CS-FOF concentra la importancia SHAP en los estados de pago (47 % frente a 39 %);
  - XGBoost reparte más entre el cupo y los montos (52 % frente a 42 %);
  - CS-FOF falla más en los **clientes al día** (F1 0.261 frente a 0.289), que son el 77 % de la cartera.

**En contra**

- Los modelos se parecen mucho: la correlación de Spearman entre sus importancias SHAP es 0.94.
- Hay un techo de información: todos rondan F1 ≈ 0.55 y AUC ≈ 0.78 con estas variables.
- Con 5 pliegues, las pruebas tienen poca potencia.

**Cuidados**

- El stacking debe ir **dentro** de cada pliegue externo de la validación anidada.
- Hay que usar `predict_proba`, un meta-modelo simple y ajustar el umbral.
- Si el ensamble incluye a CS-FOF, ya no es "el modelo original". Conviene reportarlo como un experimento
  adicional.

---

## 3. Revisión del artículo y de la literatura reciente

### 3.1 Lo que dice el artículo (PDF de 35 páginas)

- CS-FOF obtiene el mayor F1 (0.5479 ± 0.0054) y el mayor AUC-ROC (0.7847), pero **es estadísticamente
  equivalente** a XGBoost (+0.0014, IC 95 % [−0.0059, 0.0086], p_adj = 0.63) y a Random Forest.
- Supera con significancia a SVM, KNN, el árbol de decisión, las cuatro regresiones logísticas y Naive
  Bayes.
- En la ablación, el corte oblicuo aporta poco (−0.0024 sin él, no significativo). La ponderación por
  costo es lo que más aporta (−0.071 sin ella).
- La condición R3 del curso (superioridad estadística) **no se cumple**.

> **Nota:** no encontramos ningún modelo llamado "ELAR" ni en el PDF ni en los repositorios. Asumimos que
> se refería a CS-FOF.

### 3.2 Lo que dice la literatura

- **Nadie supera con claridad a un XGBoost bien ajustado.** En TabArena (NeurIPS 2025), LightGBM,
  CatBoost y XGBoost ajustados y ensamblados quedan muy cerca (Elo ≈ 1410, 1398 y 1357). Lo que más sube
  el ranking es el **ensamblado posterior**, no un algoritmo nuevo.
- **Techo del conjunto de Taiwán:** F1 ≈ 0.55 y AUC ≈ 0.78. Hay trabajos de stacking que reportan AUC
  de 0.87 a 0.93, pero probablemente tienen fuga de información o particiones distintas.
- **El obstáculo principal es la potencia estadística.** Con 5 pliegues, una ventaja tendría que pasar de
  unos **0.006 a 0.008 de F1** para ser significativa. Con 10 × 10 repeticiones baja a unos **0.003**.

### 3.3 Modelos de ensamble no neuronales recomendados

| # | Modelo | Por qué podría ganar | Riesgo |
|---|---|---|---|
| 1 | **Selección de ensambles de Caruana** (como AutoGluon y TabArena) | Es lo que más sube el Elo en TabArena | Si incluye a XGBoost, ganarle es casi esperable |
| 2 | **Stacking multicapa estilo AutoGluon** | Es el mejor sistema no neuronal en muchos benchmarks | Costoso |
| 3 | **Rotation Forest** | Bagnall et al. (2018) lo encontró significativamente mejor que XGBoost en datos continuos | Sin evidencia específica en crédito |
| 4 | **CatBoost** (*ordered boosting*) | Ya empata (0.5480) | Lo realista es un empate |

### 3.4 Mejoras propuestas para CS-FOF

**A. Diseño estadístico (lo más importante)**
1. Validación anidada **repetida** (10 repeticiones × 5 pliegues) con las mismas particiones para todos.
2. **Pre-registrar un único contraste principal** frente a XGBoost.
3. Elegir el umbral **dentro del bucle interno** para todos los modelos.

**B. Atacar la debilidad con los clientes al día**
1. Versión con **boosting** de los cortes de Fisher (CS-FOB).
2. **Varias direcciones por nodo** (Fisher + proyecciones dispersas al estilo SPORF).
3. **Regresión logística en las hojas** (*model trees*).
4. **Variables de dominio:** utilización, razón pago/factura, meses con atraso y atraso máximo. Hay que
   dárselas también a XGBoost para que la comparación sea justa.
5. Buscar el costo *c* como hiperparámetro.

**C. Stacking centrado en CS-FOF**
1. **Stacking con pesos que dependen de las variables** (FWLS, Sill et al., 2009) según PAY_0.
2. Mezcla de expertos con una compuerta fija en PAY_0.
3. Stacking clásico: CS-FOF + CatBoost + regresión logística.

**Ablación clave:** comparar **stacking con CS-FOF frente a stacking sin CS-FOF**. Es más defendible que
"el ensamble le gana a XGBoost".

### 3.5 Advertencia metodológica

El artículo declara que no se ajustó el modelo después de ver la prueba. Si ahora se modifica CS-FOF
hasta superar a XGBoost con las mismas particiones, el resultado queda contaminado. Por eso hay que:
- presentar la nueva versión como un **experimento nuevo y pre-registrado**;
- usar **semillas de partición nuevas**;
- reportar el resultado aunque sea un empate.

---

## 4. Implementación: stacking con validación repetida

### 4.1 Archivos nuevos (commit `5f0cf11`)

| Archivo | Qué hace |
|---|---|
| `src/stacking.py` | Los 4 combinadores de nivel 1 y la función que elige el umbral de máximo F1 |
| `src/run_stacking_repeated.py` | Corre la validación repetida 10 × 5 con los 6 modelos base (reanudable) |
| `src/stacking_analysis.py` | Nadeau–Bengio corregida, t bayesiana con ROPE, Wilcoxon y Holm |
| `experiments/protocolo_stacking.md` | Hipótesis y reglas fijadas antes de correr |
| `tests/test_stacking.py` | 9 pruebas nuevas (las 44 del repositorio pasan) |

### 4.2 Diseño

- **Modelos base:** CS-FOF, XGBoost, CatBoost, LightGBM, Random Forest y regresión logística, con
  ponderación de clases y el preprocesamiento del proyecto.
- **Hiperparámetros:** los de `artifacts/metadata*.json` por defecto (algo optimistas, pero iguales para
  todos), o búsqueda interna con `--busqueda N`.
- **Particiones:** 10 repeticiones × 5 pliegues, semillas **1000 a 1009**, distintas de 42 y de 1 a 4.
- **Combinadores**, cada uno con y sin CS-FOF:
  - `stack_fwls`: stacking con pesos distintos para clientes al día y atrasados (PAY_0 > 0);
  - `stack_lr`: regresión logística sobre los logits;
  - `caruana`: selección de ensambles de Caruana (optimiza AUC-PR);
  - `promedio`: promedio simple.
- **Sin fuga de información:** en cada pliegue externo, los combinadores y el umbral se ajustan solo con
  predicciones *out-of-fold* del entrenamiento.

### 4.3 Hipótesis pre-registradas (Holm sobre las dos)

- **H1:** `stack_fwls_con` tiene mayor F1 que XGBoost.
- **H2:** `stack_fwls_con` tiene mayor F1 que `stack_fwls_sin`, es decir, CS-FOF aporta información propia.

**Métrica principal:** F1 de la clase *default* con el umbral elegido en el entrenamiento (`f1_umbral`).
**Secundarias:** F1 con umbral 0.5, precisión, *recall*, MCC, AUC-ROC y AUC-PR.

**Regla de decisión:** se confirma si p_Holm < 0.05 y la diferencia es positiva. Si H1 se confirma y H2
no, el ensamble supera a XGBoost **sin** evidencia de que CS-FOF sea el responsable.

---

## 5. Cómo correrlo

Desde la raíz del repositorio, con el entorno `creditcard` activado.

**1. Generar el dataset**, si no existe `data/processed/dataset_final.csv`. Antes hay que poner el `.xls`
de UCI en `data/raw/default_of_credit_card_clients.xls`.

```bash
python -m src.preprocessing
```

**2. Congelar el protocolo antes de correr**, para que el pre-registro quede demostrable.

```bash
git tag protocolo-stacking-v1
```

```bash
git push origin protocolo-stacking-v1
```

**3. Correr el estudio.** Ajusten `CSFOF_N_JOBS` al número de núcleos.

```bash
CSFOF_N_JOBS=4 python -u -m src.run_stacking_repeated
```

En Windows (PowerShell): `$env:CSFOF_N_JOBS=4; python -u -m src.run_stacking_repeated`

**4. Analizar los resultados.**

```bash
python -m src.stacking_analysis
```

**Tiempo:** unos 111 s por pliegue con 4 núcleos (CS-FOF tarda 58 s de esos), es decir, **unas 1.5 horas**
para las 50 particiones. Si se interrumpe, al relanzar el mismo comando sigue desde donde quedó.

**Opciones útiles**

| Opción | Para qué |
|---|---|
| `--muestra 3000 --repeticiones 1 --salida results/stacking_prueba` | Prueba rápida (unos 40 s) |
| `--busqueda 25` | Protocolo estricto con búsqueda interna (mucho más lento) |
| `--metrica f1_05` (en el análisis) | Repetir el análisis con umbral 0.5, como en el artículo |

### 5.1 Qué genera y cómo leerlo

- `results/stacking_repeated/metricas.csv`: una fila por repetición, pliegue y candidato.
- `resumen.csv`: media y desviación de cada candidato.
- `contrastes.csv`: diferencia de F1, IC 95 %, p de Nadeau–Bengio ajustado por Holm, probabilidades
  bayesianas (peor / equivalente / mejor, ROPE ±0.01) y la columna `significativo`.

Si `significativo` es `True` y la diferencia es positiva en H1, el ensamble supera a XGBoost. Si además
lo es en H2, parte de la mejora se puede atribuir a CS-FOF.

### 5.2 Advertencias

- **Hiperparámetros algo optimistas:** los de `artifacts/` se eligieron con todos los datos; quedó
  declarado como limitación en el protocolo.
- **Prueba de tiempo previa:** se corrió un solo pliegue con los datos completos para medir el tiempo y
  se detuvo ahí. Quedó anotado en la sección "Desviaciones" del protocolo; no se cambió nada después.
- **El resultado puede ser un empate:** este diseño solo detecta mejoras de unos 0.003 de F1 o más. Si
  sale empate, se reporta igual.

---

## 6. Fuentes

- [TabArena (NeurIPS 2025), nota del artículo](https://en.papernotes.org/NeurIPS2025/self_supervised/tabarena_a_living_benchmark_for_machine_learning_on_tabular_data/)
- [EXAONE-Tabular, tabla de Elo de TabArena](https://huggingface.co/LG-AI-Research/EXAONE-Tabular)
- [A Closer Look at Deep Learning Methods on Tabular Datasets](https://www.lamda.nju.edu.cn/caihr/file/tabular_benchmark/paper.pdf)
- [Light-Weight Benchmarks… Tabular Foundation Models (2025)](https://arxiv.org/html/2512.00888v1)
- [Is rotation forest the best classifier for problems with continuous features? (Bagnall et al.)](https://arxiv.org/pdf/1809.06705)
- [Consistency of the oblique decision tree and its boosting and random forest (Bernoulli, 2026)](https://projecteuclid.org/journals/bernoulli/volume-32/issue-1/Consistency-of-the-oblique-decision-tree-and-its-boosting-and/10.3150/25-BEJ1864.short)
- [Supplement: oblique boosting vs XGB](https://arxiv.org/pdf/2211.12653)
- [ODRF R package (R Journal, 2026)](https://journal.r-project.org/articles/RJ-2026-041/)
- [Stacked classifier for credit risk (Journal of Big Data, 2024)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-024-00882-0)
- [Credit card default prediction: empirical analysis](https://www.researchgate.net/publication/391596843_Credit_Card_Default_Prediction_An_Empirical_Analysis_on_Predictive_Performance_Using_Statistical_and_Machine_Learning_Methods)
- [Comparative Study of Credit Default Risk Modelling with Ensemble Methods](https://link.springer.com/chapter/10.1007/978-3-032-23897-9_22)
- [Inherently Interpretable Tree Ensemble Learning (2024)](https://arxiv.org/pdf/2410.19098)
- [Predicting credit default probabilities with unequal class distributions](https://arxiv.org/pdf/1907.12996)
