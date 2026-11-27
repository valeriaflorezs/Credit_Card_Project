# Predicción del Incumplimiento de Pago en Tarjetas de Crédito usando Machine Learning

<div class="cc-hero">
<div class="cc-card">
<div class="cc-card-top"><span>Riesgo de crédito</span><span>&#9678;</span></div>
<div class="cc-chip-ic"></div>
<div class="cc-number">&bull;&bull;&bull;&bull; &nbsp;&bull;&bull;&bull;&bull; &nbsp;&bull;&bull;&bull;&bull; &nbsp;2005</div>
<div class="cc-card-bottom"><div><small>Titulares</small><strong>Katherin Barrera</strong><br><strong>Valeria Florez</strong></div><div style="text-align:right"><small>Periodo de datos</small><strong>Abr 2005 a Sep 2005</strong></div></div>
</div>
<div class="cc-hero-text">
<p class="cc-kicker">Proyecto de análisis y modelado</p>
<p class="cc-names">Katherin Barrera López y Valeria Florez Sarmiento</p>
<p>Departamento de Matemáticas, Física y Ciencia de Datos<br>Universidad del Norte, Barranquilla, Colombia</p>
<p>lkatherin@uninorte.edu.co<br>florezvaleria@uninorte.edu.co</p>
<a class="cc-btn" href="https://dash-credit-kv.onrender.com">Abrir el tablero interactivo</a>
</div>
</div>

::::{grid} 2 2 4 4
:gutter: 3

:::{grid-item-card}
:class-card: cc-kpi

<p class="kpi-label">Clientes analizados</p>
<p class="kpi-value">29,965</p>
<p class="kpi-note">tras eliminar 35 duplicados</p>
:::

:::{grid-item-card}
:class-card: cc-kpi

<p class="kpi-label">Tasa de default</p>
<p class="kpi-value">22.1 %</p>
<p class="kpi-note">6,630 clientes</p>
:::

:::{grid-item-card}
:class-card: cc-kpi

<p class="kpi-label">Combinaciones evaluadas</p>
<p class="kpi-value">116</p>
<p class="kpi-note">112 del diseño factorial + 4 de CS-FOF</p>
:::

:::{grid-item-card}
:class-card: cc-kpi

<p class="kpi-label">Mayor F1 (validación anidada 5 × 5)</p>
<p class="kpi-value">0.548</p>
<p class="kpi-note">CS-FOF 0.5479; XGBoost 0.5466. Diferencia no significativa</p>
:::
::::

## Resumen

:::{div} cc-box cc-resumen
El incumplimiento de pago en tarjetas de crédito genera pérdidas, costos de cobranza y exigencias de capital para las
entidades financieras; anticiparlo permite ajustar cupos, priorizar la cobranza y reducir la exposición al riesgo. En
el conjunto de datos de clientes de Taiwán se han usado modelos estadísticos, ensambles de árboles, remuestreo y redes
profundas; sus resultados son muy similares, las redes pierden interpretabilidad y los árboles solo dividen el espacio
en paralelo a los ejes. Se propone el *Cost-Sensitive Fisher Oblique Forest* (CS-FOF), un bosque aleatorio
implementado desde cero cuyos nodos pueden cortar por la dirección discriminante de Fisher ponderada por el costo del
incumplimiento, y se compara con 112 combinaciones de 7 modelos de referencia mediante validación cruzada anidada 5 × 5 y un
protocolo de pruebas de hipótesis. CS-FOF obtiene el mayor F1 de la clase *default*, 0.5479 ± 0.0054, frente a
0.5466 de XGBoost, el mejor modelo de referencia, con el mayor AUC-ROC (0.785) y la menor variabilidad de los tres
mejores, pero **no es superior de forma estadísticamente significativa** a XGBoost ni a Random Forest, y no es
consistente en tres conjuntos de crédito adicionales. Tiene el mayor F1 con etiquetas ruidosas y con muestras
moderadas. El
aporte es un clasificador no neuronal, con tres propiedades demostradas, que iguala a los mejores modelos de referencia en este
problema y se reporta con un veredicto honesto sobre su superioridad.

<div class="cc-chips"><span class="cc-chip">riesgo crediticio</span><span class="cc-chip">incumplimiento de pago</span><span class="cc-chip">tarjetas de crédito</span><span class="cc-chip">árboles de decisión oblicuos</span><span class="cc-chip">bosques aleatorios</span><span class="cc-chip">discriminante de Fisher</span><span class="cc-chip">aprendizaje sensible al costo</span><span class="cc-chip">desbalance de clases</span></div>
:::

## Resultados en cifras

| Modelo | F1 de la clase *default* | Validación |
|---|---|---|
| Regresión logística (línea base) | cerca de 0.38 | Partición 80/20 |
| Mejor modelo de referencia (XGBoost, pesos de clase, Optuna) | 0.5466 ± 0.0091 | Anidada 5 × 5 |
| **CS-FOF** (modelo original, pesos de clase, búsqueda aleatoria) | **0.5479 ± 0.0054** | Anidada 5 × 5 |
| **HIB** (ensamble de 8 modelos a partir de CS-FOF; XGBoost en la misma corrida: 0.5468) | **0.5505** | 5 × 5 con semillas nuevas |

:::{admonition} Qué se concluye
:class: cc-limites

- CS-FOF obtiene el mayor F1 medio, pero la diferencia no es significativa: Friedman detecta diferencias entre los 8 modelos ($\chi^2 = 32.9$), pero Nadeau–Bengio, Wilcoxon y DeLong no separan a CS-FOF de XGBoost ni de Random Forest. La condición R3 de superioridad **no se cumple**.
- CS-FOF sí supera con significancia y efecto grande a SVM, KNN, árbol de decisión, regresión logística y Naive Bayes.
- En la evaluación ampliada tiene el mayor F1 con ruido en las etiquetas y con muestras moderadas, pero falla más que XGBoost con los clientes al día en septiembre y con desbalance extremo (Polish).
- Para mejorar el F1 se hizo un ensamble (*stacking*) a partir de CS-FOF con XGBoost, CatBoost, LightGBM, Random Forest, Extra Trees, HistGradientBoosting y regresión logística: HIB gana a los 7 modelos base con Wilcoxon con Holm, DeLong y McNemar (+0.0037 de F1 sobre XGBoost), pero Nadeau–Bengio no rechaza frente a XGBoost.
- Los mejores modelos rondan F1 ≈ 0.55 y AUC ≈ 0.78 y quedan muy cerca entre sí; eso no prueba un techo del problema, pero hace difícil separarlos estadísticamente.
:::

## Contenido del libro

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} 0. Introducción
:link: introduccion
:link-type: doc
:class-card: cc-step sd-card-hover

Problema, hipótesis, objetivos, marco teórico y metodología.
:::

:::{grid-item-card} 1. Trazabilidad de correcciones
:link: correcciones
:link-type: doc
:class-card: cc-step sd-card-hover

Observaciones del Entregable 2, acciones realizadas y auditoría de fuga.
:::

:::{grid-item-card} 2. Datos: ETL y EDA
:link: notebooks/00_eda
:link-type: doc
:class-card: cc-step sd-card-hover

Análisis exploratorio, dataset final y declaración de los datos.
:::

:::{grid-item-card} 3. Modelos benchmark
:link: notebooks/05_results_analysis
:link-type: doc
:class-card: cc-step sd-card-hover

Las 112 combinaciones, interpretabilidad, tabla maestra, curvas anytime y validación del genético.
:::

:::{grid-item-card} 4. Literatura y novedad
:link: literatura
:link-type: doc
:class-card: cc-step sd-card-hover

Matriz de búsqueda en tres bases de datos y declaración de novedad.
:::

:::{grid-item-card} 5. Modelo original
:link: modelo_original
:link-type: doc
:class-card: cc-step sd-card-hover

CS-FOF: formulación, pseudocódigo, propiedades, código, pruebas, ablación y SHAP, y su mejora con el ensamble HIB.
:::

:::{grid-item-card} 6. Validación estadística
:link: validacion
:link-type: doc
:class-card: cc-step sd-card-hover

Protocolo, pruebas de hipótesis, tamaños del efecto y veredicto.
:::

:::{grid-item-card} 7. Evaluación ampliada
:link: evaluacion_ampliada
:link-type: doc
:class-card: cc-step sd-card-hover

Conjuntos adicionales, robustez, calibración, incertidumbre y análisis de errores.
:::

:::{grid-item-card} 8. Dashboard y despliegue
:link: dashboard
:link-type: doc
:class-card: cc-step sd-card-hover

Tablero en Render, predicción en vivo, ficha del modelo y artefactos.
:::

:::{grid-item-card} 9. Conclusiones
:link: conclusiones
:link-type: doc
:class-card: cc-step sd-card-hover

Hallazgos, veredicto sobre R1, R2 y R3, limitaciones y trabajo futuro.
:::

:::{grid-item-card} 10. Reproducibilidad
:link: reproducibilidad
:link-type: doc
:class-card: cc-step sd-card-hover

Entorno, semillas, lista de verificación y declaración de autoría.
:::
::::
