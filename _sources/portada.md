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

<p class="kpi-label">Configuraciones</p>
<p class="kpi-value">112</p>
<p class="kpi-note">7 modelos x 4 balanceos x 4 optimizadores</p>
:::

:::{grid-item-card}
:class-card: cc-kpi

<p class="kpi-label">Mejor F1 de las 112</p>
<p class="kpi-value">0.547</p>
<p class="kpi-note">XGBoost, validación anidada 5 x 5</p>
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
incumplimiento; se basa en los árboles oblicuos y se evalúa con validación cruzada anidada y pruebas de hipótesis. En
un experimento piloto, CS-FOF obtuvo el mayor F1 de la clase *default* entre seis modelos (0.547 frente a 0.546, una
mejora relativa de 0.3%) con un AUC-ROC de 0.780; se espera que la validación final confirme esta ventaja de forma
significativa. El aporte es un clasificador interpretable y no neuronal que lleva los cortes oblicuos al riesgo
crediticio, con garantías teóricas sobre la calidad de cada partición.

<div class="cc-chips"><span class="cc-chip">riesgo crediticio</span><span class="cc-chip">incumplimiento de pago</span><span class="cc-chip">tarjetas de crédito</span><span class="cc-chip">árboles de decisión oblicuos</span><span class="cc-chip">bosques aleatorios</span><span class="cc-chip">discriminante de Fisher</span><span class="cc-chip">aprendizaje sensible al costo</span><span class="cc-chip">desbalance de clases</span></div>
:::

## Resultados en cifras

| Modelo | F1 de la clase *default* |
|---|---|
| Regresión logística (línea base, partición 80/20) | cerca de 0.38 |
| Mejores combinaciones con balanceo (validación anidada 5 x 5) | entre 0.52 y 0.55 |
| XGBoost con `class_weight` y optimización bayesiana | 0.547 |

:::{admonition} Qué se concluye
:class: cc-limites

- La mejora sobre la línea base proviene en buena parte del balanceo, que desplaza el umbral de decisión; el ROC-AUC casi no cambia entre técnicas.
- Con 5 folds, la prueba post-hoc de Nemenyi solo separa combinaciones muy alejadas en el ranking (224 de 6,216 pares); entre las combinaciones de cabeza no hay diferencias significativas, por lo que esa comparación es descriptiva.
- La interpretabilidad se aborda con SHAP en los 5 folds (Random Forest y XGBoost) y una comparación LIME frente a SHAP.
:::

## Contenido del libro

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} 1. Marco del proyecto
:link: introduccion
:link-type: doc
:class-card: cc-step sd-card-hover

Introducción, problema, hipótesis, objetivos, marco teórico y metodología.
:::

:::{grid-item-card} 2. Análisis exploratorio
:link: notebooks/00_eda
:link-type: doc
:class-card: cc-step sd-card-hover

EDA y construcción del dataset final.
:::

:::{grid-item-card} 3. Modelo base
:link: notebooks/01_baseline_logistic
:link-type: doc
:class-card: cc-step sd-card-hover

Regresión logística con partición 80/20.
:::

:::{grid-item-card} 4. Pipeline y experimentos
:link: notebooks/03_pipeline_dev
:link-type: doc
:class-card: cc-step sd-card-hover

Desarrollo del pipeline y corrida de las combinaciones.
:::

:::{grid-item-card} 5. Análisis de resultados
:link: notebooks/05_results_analysis
:link-type: doc
:class-card: cc-step sd-card-hover

Tabla de las 112 combinaciones, comparación estadística, calibración y complejidad.
:::

:::{grid-item-card} 6. Interpretabilidad
:link: notebooks/06_interpretability
:link-type: doc
:class-card: cc-step sd-card-hover

SHAP y LIME.
:::

:::{grid-item-card} 7. Modelo original
:link: notebooks/10_modelo_original_cs_fof
:link-type: doc
:class-card: cc-step sd-card-hover

CS-FOF, LightGBM y CatBoost frente al mejor F1 de las 112 combinaciones.
:::
::::
