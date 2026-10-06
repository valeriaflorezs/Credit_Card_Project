# Introducción y marco del proyecto

## Introducción

La predicción del incumplimiento de pago (*credit default*) es uno de los problemas de mayor relevancia
práctica en la gestión del riesgo crediticio, por su efecto directo sobre la solvencia de las entidades
financieras y sobre las políticas de otorgamiento de crédito. Anticipar qué clientes tienen mayor probabilidad
de caer en mora permite ajustar límites de crédito, diferenciar tasas de interés y diseñar estrategias de
cobranza preventiva, reduciendo la exposición a pérdidas {cite:p}`yeh2009`.

En aprendizaje automático el problema se aborda como una clasificación binaria supervisada, con modelos que van
desde la regresión logística, interpretable, hasta métodos de ensamble como Random Forest y *gradient
boosting*, cuyo desempeño comparativo se evaluó originalmente sobre el conjunto de datos de este trabajo
{cite:p}`yeh2009`. Técnicas de interpretabilidad como SHAP {cite:p}`lundberg2017` complementan esos modelos con
explicaciones por variable, algo valioso en el sector financiero, donde la trazabilidad de las decisiones
automatizadas suele ser un requisito.

## Planteamiento del problema

El conjunto de datos contiene 30,000 clientes de tarjetas de crédito en Taiwán (2005), con 23 variables
predictoras (límite de crédito, datos demográficos, historial de pagos, montos facturados y montos pagados de
seis meses) y una variable objetivo binaria: si el cliente incumplió el pago al mes siguiente.

Tres dificultades hacen que el problema no se resuelva con un modelo estándar:

- **Desbalance de clases.** Solo ≈22 % de los clientes incumple. Un modelo entrenado sin ajustes favorece la
  clase mayoritaria y puede mostrar un *accuracy* alto mientras detecta pocos casos de *default*. La regresión
  logística base de este proyecto, por ejemplo, tiene un F1 de la clase minoritaria de ≈0.38.
- **Calidad y estructura de los datos.** Hay duplicados exactos, categorías no documentadas en `EDUCATION` y
  `MARRIAGE`, códigos ambiguos en las variables `PAY_x`, distribuciones muy asimétricas y una multicolinealidad
  severa entre las seis variables `BILL_AMT`.
- **Muchas decisiones que interactúan.** El algoritmo, la técnica de balanceo y la forma de optimizar los
  hiperparámetros afectan el resultado y rara vez se comparan de manera conjunta y con una validación que
  evite la fuga de información.

Para una entidad financiera el costo de equivocarse no es simétrico. Si concede cupos altos a clientes que luego
incumplen, pierde capital, aumentan sus costos de cobranza y debe constituir mayores provisiones. Si restringe
el crédito en exceso, pierde ingresos por intereses y comisiones. En esta muestra 6,630 clientes (22.1 %) no
pagaron su tarjeta el mes siguiente, y no detectar a quien incumplirá (falso negativo) suele costar más que
revisar de más a quien sí pagaría (falso positivo).

### Preguntas de investigación

La pregunta general es: **¿qué características del cliente, de su línea de crédito y de su historial reciente de
pagos se asocian con el incumplimiento del mes siguiente, y con qué intensidad?** A partir de ella, el trabajo
se pregunta:

- ¿Cómo se distribuyen las variables y qué problemas de calidad presentan?
- ¿Qué tan desbalanceada está la variable objetivo?
- ¿Qué variables separan mejor a los clientes que incumplen?
- ¿Existe redundancia entre las variables predictoras?
- ¿Qué combinación de algoritmo, técnica de balanceo y método de optimización detecta mejor a los clientes en
  riesgo de incumplimiento, y qué parte de la mejora se debe a cada decisión?

### Hipótesis exploratorias

| | Hipótesis |
|---|---|
| H1 | El historial reciente de pagos (`PAY_0` … `PAY_6`) es el factor más fuertemente asociado al incumplimiento. |
| H2 | Los clientes con menor límite de crédito (`LIMIT_BAL`) presentan una mayor tasa de *default*. |
| H3 | Las variables demográficas (sexo, educación, estado civil, edad) se asocian con el *default*, pero con un tamaño de efecto pequeño. |
| H4 | Los montos facturados mensuales (`BILL_AMT1–6`) están fuertemente correlacionados entre sí (multicolinealidad). |
| H5 | La clase *default* es minoritaria, por lo que el problema presenta desbalance de clases. |

### Alcance y limitaciones

- Los datos provienen de un solo banco de Taiwán y de un periodo corto (abril a septiembre de 2005), por lo que
  no se generalizan sin cautela a otros mercados o épocas.
- El análisis es asociativo: una correlación no implica causalidad.
- Los códigos `-2` y `0` de `PAY_x` no están documentados con precisión en la fuente y se conservan sin
  recodificar.
- Las cifras monetarias están en dólares taiwaneses (NT$).

## Justificación

En riesgo crediticio un falso negativo, un cliente que incumple y no fue detectado, es más costoso que un
falso positivo. Por eso el *accuracy* no basta, y el trabajo prioriza métricas sensibles a la clase minoritaria
(F1, *recall*, PR-AUC). Documentar de forma reproducible el preprocesamiento y comparar de manera sistemática
las alternativas permite separar el aporte real de cada decisión (por ejemplo, el balanceo frente al modelo
elegido) y evitar conclusiones basadas en una sola configuración.

## Objetivos

### Objetivo general

Evaluar de forma sistemática el efecto del preprocesamiento, el algoritmo, la técnica de balanceo y el método de
optimización de hiperparámetros sobre la capacidad de predecir el incumplimiento de pago de clientes de tarjetas
de crédito.

### Objetivos específicos

1. Describir el problema y el conjunto de datos, incluida la variable objetivo y su nivel de desbalance.
2. Documentar de forma reproducible el proceso ETL, justificando cada decisión de limpieza.
3. Caracterizar estadísticamente las variables mediante un EDA univariado y bivariado, y evaluar la
   multicolinealidad.
4. Entrenar un modelo base (regresión logística) sobre una partición *train/test* aislada de fuga de
   información, y cuantificar su margen de mejora, en especial en falsos negativos y falsos positivos.
5. Comparar 112 configuraciones (7 modelos × 4 técnicas de balanceo × 4 métodos de optimización) con
   validación cruzada anidada y el F1 de la clase *default* como métrica principal.
6. Contrastar estadísticamente las configuraciones y analizar la calibración de probabilidades y el costo
   computacional.
7. Interpretar los modelos con SHAP y LIME para identificar las variables que más influyen en la predicción.

## Marco teórico

**Riesgo de crédito y *default*.** El riesgo de crédito es la posibilidad de que una entidad sufra pérdidas
porque un deudor no cumple sus obligaciones. En Basilea II la pérdida esperada se descompone como
$EL = PD \times LGD \times EAD$, donde $PD$ es la probabilidad de incumplimiento en un horizonte dado (lo que
estudian estos datos), $LGD$ la fracción de la exposición que se pierde si hay incumplimiento y $EAD$ el monto
expuesto en ese momento.

**Scoring crediticio.** Los modelos de *scoring* estiman la $PD$ a partir del comportamiento de pago, el
límite, la deuda y el perfil demográfico, y sirven para aprobar solicitudes, fijar cupos y priorizar la
cobranza.

**Análisis exploratorio de datos (EDA).** Propuesto por Tukey {cite:p}`tukey1977`: explorar los datos con
resúmenes numéricos y gráficos antes de ajustar modelos, para descubrir patrones, anomalías y supuestos.

**Valores atípicos y asimetría.** Se usa el criterio de Tukey: es atípico un valor menor que $Q_1 - 1.5\,IQR$ o
mayor que $Q_3 + 1.5\,IQR$. La winsorización recorta los extremos a percentiles sin eliminar clientes, y la
transformación logarítmica con signo suaviza colas largas cuando hay ceros o negativos.

**Desbalance de clases.** Con una clase minoritaria, un clasificador puede lograr alta exactitud ignorándola.
Por eso se pondera las clases (`class_weight`), se sobremuestrea la minoritaria con SMOTE {cite:p}`chawla2002`
o ADASYN, se ajusta el umbral y se evalúa con F1, *recall* de la clase minoritaria y AUC en lugar de *accuracy*.

**Correlación de Spearman y multicolinealidad.** Spearman {cite:p}`spearman1904` mide asociación monótona con
los rangos de los datos, sin exigir linealidad ni normalidad, y resiste los valores extremos. Se prefiere sobre
Pearson porque los montos son muy asimétricos, las variables `PAY_x` son ordinales y el objetivo es binario. La
multicolinealidad se mide con $VIF_j = 1/(1 - R_j^2)$; un VIF mayor que 10 (algunos autores usan 5) es señal de
alerta.

**Asociación y comparación de grupos.** La prueba $\chi^2$ de independencia y la V de Cramér
{cite:p}`cramer1946` miden la asociación entre variables categóricas. Para comparar grupos sin suponer
normalidad se usan Mann–Whitney {cite:p}`mann1947` y Kruskal–Wallis {cite:p}`kruskal1952`, y la t de Welch
para medias con varianzas distintas. Con $n \approx 30\,000$ casi cualquier diferencia resulta significativa,
así que se reporta también el tamaño del efecto.

**Antecedentes.** Yeh y Lien {cite:p}`yeh2009` compararon seis técnicas de minería de datos (regresión
logística, análisis discriminante, k vecinos, Naive Bayes, redes neuronales y árboles de clasificación) con
estos mismos datos y propusieron un método de suavizado por ordenamiento para estimar la probabilidad real de
incumplimiento. Este trabajo parte de ese conjunto de datos, completa la etapa de comprensión de los datos y
compara de manera sistemática algoritmos modernos, técnicas de balanceo y métodos de optimización.

### Diccionario de variables

| Variable | Tipo | Descripción |
|---|---|---|
| `LIMIT_BAL` | Numérica continua | Monto del crédito otorgado (individual y familiar) en NT$. |
| `SEX` | Categórica | 1 = hombre, 2 = mujer. |
| `EDUCATION` | Categórica ordinal | 1 = posgrado, 2 = universidad, 3 = secundaria, 4 = otros (los códigos 0, 5 y 6 no documentados se reagrupan en "Otros"). |
| `MARRIAGE` | Categórica | 1 = casado, 2 = soltero, 3 = otros (el código 0 se reagrupa en "Otros"). |
| `AGE` | Numérica discreta | Edad en años. |
| `PAY_0`, `PAY_2`–`PAY_6` | Ordinal | Estado de pago mensual, de septiembre (`PAY_0`) a abril (`PAY_6`). −2, −1 y 0 = sin atraso; 1 = un mes de atraso; y así hasta 8–9 meses. El dataset no incluye `PAY_1`. |
| `BILL_AMT1`–`BILL_AMT6` | Numérica continua | Monto facturado de septiembre (1) a abril (6), en NT$. Puede ser negativo (saldo a favor). |
| `PAY_AMT1`–`PAY_AMT6` | Numérica continua | Monto efectivamente pagado de septiembre (1) a abril (6), en NT$. |
| `default_payment_next_month` | Binaria (objetivo) | 1 = el cliente incumple el pago el mes siguiente (octubre de 2005); 0 = no incumple. |

## Metodología

### Datos

*Default of Credit Card Clients Dataset* del repositorio UCI {cite:p}`uci`, publicado por Yeh y Lien
{cite:p}`yeh2009`: 30,000 clientes, 23 variables predictoras y la variable objetivo
`default_payment_next_month` (≈78 % clase 0, ≈22 % clase 1).

### ETL

- **Extract:** lectura del archivo fuente con `pandas`, renombrado de la variable objetivo y uso de `ID` solo
  como índice, no como predictor.
- **Transform:** eliminación de 35 duplicados exactos (para que una fila no quede en entrenamiento y su copia en
  prueba); reagrupación de los códigos no documentados de `EDUCATION` y `MARRIAGE` en "Otros"; conservación de
  los códigos `-2`, `-1` y `0` de `PAY_x`, con su ambigüedad documentada como limitación; transformación
  logarítmica con signo de `LIMIT_BAL` y `PAY_AMT1-6`; reemplazo de `BILL_AMT1-6` por su promedio, su
  tendencia y el número de meses con saldo a favor, para resolver la multicolinealidad.
- **Load:** `dataset_final.csv` con 29,965 filas × 21 columnas. El escalado y los demás pasos que dependen de
  estadísticos de la muestra se aplican dentro de cada pipeline, después de la partición, para evitar fuga de
  información.

### Análisis exploratorio

Estadística descriptiva, diagnóstico de nulos y duplicados, forma de las distribuciones, valores atípicos
(criterio de Tukey), comparación formal de grupos con pruebas no paramétricas y de independencia, y
multicolinealidad con correlaciones y VIF.

### Modelo base

Regresión logística en un *pipeline* de `scikit-learn` con partición estratificada 80/20, codificación y
escalado, evaluada con *accuracy*, *log-loss*, ROC-AUC, matriz de confusión y reporte de clasificación.

### Diseño experimental

Diseño factorial de **7 × 4 × 4 = 112 configuraciones**:

| Factor | Niveles |
|---|---|
| Modelo (7) | Regresión logística, árbol de decisión, Random Forest, XGBoost, K-Nearest Neighbors, Naive Bayes gaussiano, SVM |
| Balanceo (4) | Sin balanceo, SMOTE, ADASYN, `class_weight` |
| Optimización de hiperparámetros (4) | Grid search, random search, optimización bayesiana (Optuna), algoritmo genético (DEAP) |

- **Validación:** validación cruzada anidada, con 3 folds externos para estimar el desempeño y 3 internos para
  seleccionar hiperparámetros, y `random_state=42`.
- **Métrica principal:** F1 de la clase *default*. Se reportan también precisión, *recall*, ROC-AUC y PR-AUC.
- **Ejecución:** seis modelos (96 combinaciones) se corrieron localmente. SVM (16 combinaciones) se corrió en
  Google Colab con GPU, con otro solver y otras versiones de librerías, por lo que no es estrictamente
  comparable con el resto, en especial en tiempos.
- **Comparación estadística:** prueba de Friedman con post-hoc de Nemenyi, comparación por factor y prueba de
  DeLong. Con solo 3 folds el post-hoc no puede declarar diferencias significativas, y la comparación se
  interpreta de manera descriptiva.
- **Calibración y complejidad:** curvas de calibración, error de calibración esperado (ECE) y costo
  computacional de cada configuración.

### Interpretabilidad

SHAP sobre el Random Forest y comparación LIME vs. SHAP para XGBoost sobre una observación.

## Estructura del libro

| Capítulo | Contenido |
|---|---|
| `00_eda` | EDA y construcción del dataset final |
| `01_baseline_logistic` | Modelo base de regresión logística |
| `02_setup_check`, `03_pipeline_dev` | Verificación del entorno y desarrollo del pipeline |
| `04_run_experiments` | Corrida de las combinaciones |
| `05_results_analysis` | Resultados, comparación estadística, calibración y complejidad |
| `06_interpretability` | SHAP y LIME |

## Referencias

```{bibliography}
:filter: docname in docnames
```
