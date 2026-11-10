# Marco del proyecto

## Introducción

Las tarjetas de crédito son una de las líneas de la banca minorista con mayor exposición al riesgo: cada cliente
recibe un cupo (límite de crédito) y se espera que lo pague en las fechas pactadas. Cuando no lo hace, la entidad
enfrenta pérdidas, mayores costos de cobranza y la necesidad de provisionar capital. Anticipar qué clientes tienen
mayor probabilidad de incurrir en mora permite ajustar límites de crédito, definir tasas diferenciadas y diseñar
estrategias de cobranza preventivas {cite:p}`yeh2009`.

Desde la perspectiva técnica, el problema se ha abordado extensamente como una tarea de clasificación binaria, con
modelos que van desde la regresión logística hasta métodos de ensamble como Random Forest y *gradient boosting*,
evaluados originalmente sobre el conjunto de datos de este trabajo {cite:p}`yeh2009`. Técnicas de interpretabilidad
como SHAP {cite:p}`lundberg2017` complementan estos modelos con explicaciones a nivel de variable, algo valioso en
contextos regulados como el financiero.

Este libro presenta el análisis del conjunto *Default of Credit Card Clients* (UCI), con 30,000 clientes de un banco
de Taiwán observados entre abril y septiembre de 2005. Para cada cliente se conoce su límite de crédito, sus
características demográficas, su historial de pagos de seis meses, los montos facturados, los montos pagados y si
incumplió el pago del mes siguiente (variable objetivo). El proyecto se desarrolló en etapas: comprensión del
problema, proceso ETL, análisis exploratorio (EDA), modelo base de regresión logística, comparación sistemática de
112 configuraciones de modelos, análisis de resultados e interpretabilidad. El resumen de resultados está en la
portada y cada etapa se documenta en los capítulos siguientes.

::::{grid} 2 2 4 4
:gutter: 3

:::{grid-item-card}
:class-card: cc-kpi

<p class="kpi-label">Clientes analizados</p>
<p class="kpi-value">29,965</p>
<p class="kpi-note">tras la limpieza</p>
:::

:::{grid-item-card}
:class-card: cc-kpi

<p class="kpi-label">Columnas finales</p>
<p class="kpi-value">21</p>
<p class="kpi-note">20 predictoras y el objetivo</p>
:::

:::{grid-item-card}
:class-card: cc-kpi

<p class="kpi-label">Tasa de default</p>
<p class="kpi-value">22.1 %</p>
<p class="kpi-note">6,630 clientes</p>
:::

:::{grid-item-card}
:class-card: cc-kpi

<p class="kpi-label">Periodo</p>
<p class="kpi-value">2005</p>
<p class="kpi-note">abril a septiembre, banco de Taiwán</p>
:::
::::

## Planteamiento del problema

Una entidad financiera otorga crédito sin saber con certeza quién pagará. Si concede cupos demasiado altos a
clientes que luego incumplen, la cartera se deteriora: se pierde el capital no recuperado, aumentan los costos de
cobranza y se deben constituir mayores provisiones. Si, por el contrario, restringe el crédito en exceso, pierde
ingresos por intereses y comisiones.

En esta muestra, el 22.1 % de los clientes (6,630) no pagó su tarjeta el mes siguiente. Identificar tempranamente
qué perfiles y comportamientos anticipan ese incumplimiento es la base para fijar límites, priorizar la cobranza
preventiva y alimentar un modelo de *scoring*.

Los errores no son simétricos: no detectar a quien incumplirá (falso negativo) suele costar más que revisar de más a
quien sí pagaría (falso positivo). Por eso reportar solo el *accuracy* es engañoso: un clasificador que siempre
prediga "no *default*" ya alcanzaría 77.9 % de *accuracy* sin detectar a ningún cliente en riesgo.

Tres dificultades hacen que el problema no se resuelva con un modelo estándar:

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Desbalance de clases
:class-card: cc-step

Un modelo entrenado sin ajustes favorece la clase mayoritaria. La regresión logística base de este proyecto tiene un F1 de la clase minoritaria de cerca de 0.38.
:::

:::{grid-item-card} Calidad y estructura de los datos
:class-card: cc-step

Hay duplicados exactos, categorías no documentadas en `EDUCATION` y `MARRIAGE`, códigos ambiguos en las variables `PAY_x`, distribuciones muy asimétricas y una multicolinealidad severa entre las seis variables `BILL_AMT`.
:::

:::{grid-item-card} Muchas decisiones que interactúan
:class-card: cc-step

El algoritmo, la técnica de balanceo y la forma de optimizar los hiperparámetros afectan el resultado y rara vez se comparan de manera conjunta y con una validación que evite la fuga de información.
:::
::::

:::{admonition} Pregunta de investigación
:class: cc-pregunta

¿Qué características del cliente, de su línea de crédito y de su historial reciente de pagos se asocian con el incumplimiento del mes siguiente, y con qué intensidad?
:::

:::{admonition} Preguntas específicas
:class: tip

1. ¿Cómo se distribuyen las variables y qué problemas de calidad presentan?
2. ¿Qué tan desbalanceada está la variable objetivo?
3. ¿Qué variables separan mejor a los clientes que incumplen?
4. ¿Existe redundancia entre las variables predictoras?
5. ¿Qué combinación de algoritmo, técnica de balanceo y método de optimización de hiperparámetros detecta mejor a
   los clientes en riesgo, y qué parte de la mejora se debe a cada decisión?
:::

## Hipótesis exploratorias

Se contrastan de forma descriptiva y visual a lo largo del EDA.

| # | Hipótesis |
|---|---|
| H1 | El historial reciente de pagos (`PAY_0` a `PAY_6`) es el factor más fuertemente asociado al incumplimiento. |
| H2 | Los clientes con menor límite de crédito (`LIMIT_BAL`) presentan una mayor tasa de *default*. |
| H3 | Las variables demográficas (sexo, educación, estado civil, edad) se asocian con el *default*, pero con un tamaño de efecto pequeño. |
| H4 | Los montos facturados mensuales (`BILL_AMT1` a `BILL_AMT6`) están fuertemente correlacionados entre sí (multicolinealidad). |
| H5 | La clase *default* es minoritaria, por lo que el problema presenta desbalance de clases. |

## Justificación

:::{div} cc-box
Un EDA riguroso evita errores costosos en el modelado: revela datos mal codificados, variables redundantes, sesgos de
muestreo y desbalance antes de que afecten al modelo. Además, en riesgo crediticio el *accuracy* no basta, y el
trabajo prioriza métricas sensibles a la clase minoritaria (F1, *recall*, PR-AUC). Documentar de forma reproducible
el preprocesamiento y comparar de manera sistemática las alternativas permite separar el aporte real de cada
decisión (por ejemplo, el balanceo frente al modelo elegido) y evitar conclusiones basadas en una sola
configuración. También permite explicar a perfiles no técnicos (riesgos, cobranza, gerencia) qué señales preceden al
incumplimiento, con evidencia visual y estadística.
:::

## Objetivos

:::{admonition} Objetivo general
:class: cc-pregunta

Evaluar de forma sistemática el efecto del preprocesamiento, el algoritmo, la técnica de balanceo y el método de
optimización de hiperparámetros sobre la capacidad de predecir el incumplimiento de pago de clientes de tarjetas de
crédito, a partir de un análisis exploratorio univariado y multivariado.
:::

:::{admonition} Objetivos específicos
:class: tip

1. Describir la calidad de los datos y documentar la limpieza aplicada (duplicados, categorías no documentadas,
   saldos negativos).
2. Caracterizar la distribución de cada variable, su asimetría y la presencia de valores atípicos.
3. Cuantificar el desbalance de clases y comparar a los clientes con y sin *default* según variables demográficas,
   de crédito y de pagos.
4. Identificar relaciones entre variables y detectar multicolinealidad (correlaciones y VIF).
5. Entrenar un modelo base (regresión logística) sobre una partición *train/test* libre de fuga de información y
   cuantificar su margen de mejora, con atención a falsos negativos y falsos positivos.
6. Comparar 112 configuraciones (7 modelos por 4 técnicas de balanceo por 4 métodos de optimización) con validación
   cruzada anidada y el F1 de la clase *default* como métrica principal.
7. Contrastar estadísticamente las configuraciones, analizar la calibración de probabilidades y el costo
   computacional, e interpretar los modelos con SHAP y LIME.
:::

## Alcance y limitaciones

:::{admonition} Qué tener presente
:class: cc-limites

- Los datos provienen de un solo banco de Taiwán y de un periodo corto (de abril a septiembre de 2005): no se
  generalizan sin cautela a otros mercados o épocas.
- El análisis es asociativo; una correlación no implica causalidad.
- Los códigos `-2` y `0` de `PAY_x` no están documentados con precisión en la fuente; se conservan sin recodificar.
- Existen grupos de clientes con predictoras idénticas pero etiqueta distinta (ruido propio de la fuente), lo que
  impone un techo al desempeño de cualquier modelo con estas variables.
- Las cifras monetarias están en dólares taiwaneses (NT$).
:::

## Marco teórico

### Riesgo de crédito y *default*

El riesgo de crédito es la posibilidad de que una entidad sufra pérdidas porque un deudor no cumple, total o
parcialmente, sus obligaciones. En la regulación bancaria (Basilea II) la pérdida esperada se descompone así:

:::{div} cc-formula
$$
EL = PD \times LGD \times EAD
$$
:::

donde $PD$ es la probabilidad de incumplimiento en un horizonte dado (lo que estudian estos datos), $LGD$ la fracción
de la exposición que se pierde si hay incumplimiento y $EAD$ el monto expuesto en ese momento. En este proyecto,
*default* es la variable binaria que indica si el cliente no pagó en el mes siguiente. Los modelos de *scoring*
estiman la $PD$ a partir del comportamiento de pago, el límite, la deuda y el perfil demográfico.

### Análisis exploratorio de datos

Propuesto por Tukey {cite:p}`tukey1977`, consiste en explorar los datos con resúmenes numéricos y gráficos antes de
ajustar modelos, para descubrir patrones, anomalías y supuestos. El análisis univariado estudia una variable a la
vez; el bivariado y el multivariado, las relaciones entre dos o más variables.

### Valores atípicos y asimetría

La asimetría indica hacia dónde se alarga la cola de una distribución y la curtosis qué tan pesadas son sus colas.
Según el criterio de Tukey, un valor $x$ es atípico si

:::{div} cc-formula
$$
x < Q_1 - 1.5\,\mathrm{IQR}
\quad \text{o} \quad
x > Q_3 + 1.5\,\mathrm{IQR},
\qquad \mathrm{IQR} = Q_3 - Q_1 .
$$
:::

La winsorización recorta los extremos a percentiles (por ejemplo, P1 y P99) sin eliminar clientes. Para suavizar
colas largas cuando hay ceros o negativos se usa el logaritmo con signo:

:::{div} cc-formula
$$
\operatorname{slog}(x) = \operatorname{sign}(x)\,\ln\!\left(1 + |x|\right).
$$
:::

### Desbalance de clases

Cuando una clase es mucho menos frecuente, un clasificador puede lograr alta exactitud ignorándola. Se recomienda
ponderar las clases (`class_weight`), sobremuestrear la minoritaria (SMOTE {cite:p}`chawla2002` o ADASYN), ajustar el
umbral de decisión y evaluar con $F_1$, *recall* de la clase minoritaria y AUC en lugar del *accuracy*. Con $TP$, $FP$ y
$FN$ los verdaderos positivos, falsos positivos y falsos negativos de la clase *default*:

:::{div} cc-formula
$$
\mathrm{Precision} = \frac{TP}{TP + FP}, \qquad
\mathrm{Recall} = \frac{TP}{TP + FN}, \qquad
F_1 = \frac{2\,\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}} .
$$
:::

### Correlación de Spearman y multicolinealidad

La correlación de Spearman {cite:p}`spearman1904` mide asociación monótona: es la correlación de Pearson aplicada a
los rangos de los datos,

:::{div} cc-formula
$$
\rho_s = \rho\!\left(\operatorname{rango}(X),\, \operatorname{rango}(Y)\right).
$$
:::

No exige linealidad ni normalidad y resiste los valores extremos, lo que la hace adecuada para montos muy
asimétricos, variables `PAY_x` ordinales y un objetivo binario. Es el único método de correlación del proyecto: entre
los seis montos facturados (`BILL_AMT1` a `BILL_AMT6`) va de 0.73 a 0.91, la misma cifra del EDA y del tablero. La
multicolinealidad se evalúa con el Factor de
Inflación de la Varianza, donde $R_j^2$ es el coeficiente de determinación al regresar la variable $j$ sobre las demás
predictoras:

:::{div} cc-formula
$$
\mathrm{VIF}_j = \frac{1}{1 - R_j^2} .
$$
:::

Un VIF mayor que 10 (algunos autores usan 5) es señal de alerta.

### Asociación y comparación de grupos

La prueba $\chi^2$ de independencia contrasta si dos variables categóricas están asociadas; su tamaño de efecto se
mide con la V de Cramér {cite:p}`cramer1946`, con $n$ observaciones y $k$ el menor número de categorías entre las dos
variables:

:::{div} cc-formula
$$
V = \sqrt{\frac{\chi^2 / n}{k - 1}} ,
$$
:::

con $V < 0.1$ como efecto pequeño. Para comparar grupos sin suponer normalidad se usan Mann-Whitney
{cite:p}`mann1947` (dos grupos) y Kruskal-Wallis {cite:p}`kruskal1952` (tres o más), y la t de Welch para medias con
varianzas distintas. Con $R_g$ la suma de rangos del grupo $g$, $n_g$ su tamaño y $N$ el total de observaciones:

:::{div} cc-formula
$$
U_1 = R_1 - \frac{n_1 (n_1 + 1)}{2}, \qquad
H = \frac{12}{N(N+1)} \sum_{g} \frac{R_g^2}{n_g} - 3(N+1), \qquad
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\dfrac{s_1^2}{n_1} + \dfrac{s_2^2}{n_2}}} .
$$
:::

Con cerca de 30,000 observaciones casi cualquier diferencia resulta significativa, por lo que se reporta el tamaño
del efecto y no solo el $p$-valor.

### Calibración de probabilidades

Un modelo está calibrado si, entre los clientes a los que asigna probabilidad $p$, la proporción de *default* es
cercana a $p$. Se mide con el puntaje de Brier y con el error de calibración esperado (ECE), que agrupa las
probabilidades en $B$ intervalos:

:::{div} cc-formula
$$
\mathrm{Brier} = \frac{1}{N}\sum_{i=1}^{N}\left(\hat p_i - y_i\right)^2, \qquad
\mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{N}\,\left|\,\mathrm{acc}_b - \mathrm{conf}_b\,\right| ,
$$
:::

donde $n_b$ es el número de clientes en el intervalo $b$, $\mathrm{acc}_b$ la frecuencia observada de *default* y
$\mathrm{conf}_b$ la probabilidad media predicha.

### Antecedentes

Yeh y Lien {cite:p}`yeh2009` compararon seis técnicas de minería de datos (regresión logística, análisis
discriminante, k vecinos, Naive Bayes, redes neuronales y árboles de clasificación) con estos mismos datos y
propusieron un método de suavizado por ordenamiento para estimar la probabilidad real de incumplimiento. Este
trabajo parte de ese conjunto de datos, completa la etapa de comprensión de los datos y compara de manera
sistemática algoritmos modernos, técnicas de balanceo y métodos de optimización.

## Diccionario de variables

| Variable | Tipo | Descripción |
|---|---|---|
| `LIMIT_BAL` | Numérica continua | Monto del crédito otorgado (individual y familiar) en NT$. |
| `SEX` | Categórica | 1 = hombre, 2 = mujer. |
| `EDUCATION` | Categórica ordinal | 1 = posgrado, 2 = universidad, 3 = secundaria, 4 = otros (los códigos 0, 5 y 6 no documentados se reagrupan en "Otros"). |
| `MARRIAGE` | Categórica | 1 = casado, 2 = soltero, 3 = otros (el código 0 se reagrupa en "Otros"). |
| `AGE` | Numérica discreta | Edad en años. |
| `PAY_0`, `PAY_2` a `PAY_6` | Ordinal | Estado de pago mensual, de septiembre (`PAY_0`) a abril (`PAY_6`). -2, -1 y 0 = sin atraso; 1 = un mes de atraso; y así hasta 8 o 9 meses. El dataset no incluye `PAY_1`. |
| `BILL_AMT1` a `BILL_AMT6` | Numérica continua | Monto facturado de septiembre (1) a abril (6), en NT$. Puede ser negativo (saldo a favor). |
| `PAY_AMT1` a `PAY_AMT6` | Numérica continua | Monto efectivamente pagado de septiembre (1) a abril (6), en NT$. |
| `default_payment_next_month` | Binaria (objetivo) | 1 = el cliente incumple el pago el mes siguiente (octubre de 2005); 0 = no incumple. |

## Metodología

### Datos

*Default of Credit Card Clients Dataset* del repositorio UCI {cite:p}`uci`, publicado por Yeh y Lien
{cite:p}`yeh2009`: 30,000 clientes, 23 variables predictoras y la variable objetivo
`default_payment_next_month` (78 % clase 0, 22 % clase 1).

### ETL

- **Extract:** lectura del archivo fuente con `pandas`, renombrado de la variable objetivo y uso de `ID` solo
  como índice, no como predictor.
- **Transform:** eliminación de 35 duplicados exactos (para que una fila no quede en entrenamiento y su copia en
  prueba); reagrupación de los códigos no documentados de `EDUCATION` y `MARRIAGE` en "Otros"; conservación de
  los códigos `-2`, `-1` y `0` de `PAY_x`, con su ambigüedad documentada como limitación; transformación
  logarítmica con signo de `LIMIT_BAL` y `PAY_AMT1` a `PAY_AMT6`; reemplazo de `BILL_AMT1` a `BILL_AMT6` por su promedio, su
  tendencia y el número de meses con saldo a favor, para resolver la multicolinealidad.
- **Load:** conjunto de datos final con 29,965 filas x 21 columnas. El escalado y los demás pasos que dependen de
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

Diseño factorial de **7 x 4 x 4 = 112 configuraciones**:

| Factor | Niveles |
|---|---|
| Modelo (7) | Regresión logística, árbol de decisión, Random Forest, XGBoost, K-Nearest Neighbors, Naive Bayes gaussiano, SVM |
| Balanceo (4) | Sin balanceo, SMOTE, ADASYN, `class_weight` |
| Optimización de hiperparámetros (4) | Grid search, random search, optimización bayesiana (Optuna), algoritmo genético (DEAP) |

- **Validación:** validación cruzada anidada, con 5 folds externos para estimar el desempeño y 5 internos para
  seleccionar hiperparámetros, y `random_state=42`.
- **Métrica principal:** F1 de la clase *default*. Se reportan también precisión, *recall*, ROC-AUC y PR-AUC.
- **Ejecución:** cinco modelos (80 combinaciones) se corrieron localmente en CPU. SVM y KNN (32 combinaciones) se corrieron en
  Google Colab con una GPU T4 (cuML), con otro solver y otras versiones de librerías, por lo que no son
  estrictamente comparables con el resto, en especial en tiempos.
- **Comparación estadística:** prueba de Friedman con post-hoc de Nemenyi, comparación por factor y prueba de
  DeLong. Con 5 folds el post-hoc de Nemenyi solo separa los extremos del ranking, sin ordenar las combinaciones
  de cabeza, y la comparación entre ellas se interpreta de manera descriptiva.
- **Calibración y complejidad:** curvas de calibración, error de calibración esperado (ECE) y costo
  computacional de cada configuración.

### Modelo original y métodos recientes

Las 112 configuraciones anteriores son los modelos tradicionales y fijan el punto de comparación: su mejor F1 medio
es 0.547 (XGBoost con `class_weight` y optimización bayesiana). Con ese valor como referencia se busca un modelo que
lo supere: un bosque aleatorio propio con cortes oblicuos sensibles al costo (CS-FOF), contrastado además con dos
métodos recientes de boosting (LightGBM y CatBoost) bajo el mismo protocolo de validación anidada 5 x 5.

### Interpretabilidad

SHAP sobre el Random Forest y comparación LIME vs. SHAP para XGBoost sobre una observación.

## Estructura del libro

| Capítulo | Contenido |
|---|---|
| 00. EDA | Análisis exploratorio y construcción del conjunto de datos final |
| 01. Modelo base | Regresión logística con partición 80/20 |
| 02. Verificación del protocolo | Protocolo experimental: semilla y validación cruzada anidada |
| 03. Desarrollo del pipeline | Prueba del pipeline completo con datos sintéticos |
| 04. Experimentos | Corrida de las combinaciones |
| 05. Análisis de resultados | Resultados, comparación estadística, calibración y complejidad |
| 06. Interpretabilidad | SHAP y LIME |
| 08. Tabla maestra | Tabla maestra, comparabilidad del SVM |
| 09. Curvas anytime | Curvas anytime y multi-fidelidad |
| 10. Modelo original | CS-FOF, LightGBM y CatBoost frente a las 112 combinaciones |

## Referencias

```{bibliography}
:filter: docname in docnames
```
