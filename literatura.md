# Revisión de literatura y evidencia de novedad

Este capítulo documenta la búsqueda que respalda la condición R1 del Entregable 3: que el modelo
propuesto no se haya usado antes para predecir el incumplimiento de pago en tarjetas de crédito.

## Qué se ha usado para este problema

Una búsqueda en Web of Science (cadena 1 de la {numref}`tab-matriz-novedad`) devolvió 53 artículos sobre
el incumplimiento en tarjetas de crédito y el conjunto de datos de Taiwán. Se leyeron sus resúmenes y se
extrajeron los modelos de cada uno:

- **Modelos clásicos:** regresión logística, KNN, Naive Bayes, análisis discriminante y árboles de
  decisión, desde el trabajo original {cite:p}`yeh2009`.
- **Ensambles de árboles y *boosting*:** Random Forest, XGBoost, LightGBM, CatBoost, AdaBoost y
  *gradient boosting*, en general los de mejor ordenamiento del riesgo {cite:p}`alam2020,choi2026`.
- **Balanceo de clases:** SMOTE y sus variantes, submuestreo y regresión penalizada para eventos raros
  {cite:p}`ogundimu2019`.
- **Redes neuronales:** MLP, LSTM y *transformers* {cite:p}`hu2026,chab2026`, que no superan de forma
  consistente a los árboles en datos tabulares y pierden interpretabilidad.
- **Interpretabilidad:** SHAP sobre modelos de *boosting* {cite:p}`lundberg2017`.

El cruce completo de modelos está en `results/literatura/cruce_modelos.csv`.

## Vacío identificado

Todos los ensambles de árboles usados en este problema dividen el espacio **solo en paralelo a los
ejes**. Las variables del conjunto vienen en bloques muy correlacionados (seis estados de pago, seis
facturas y seis pagos), y una frontera diagonal entre ellas, como «el atraso crece de un mes a otro»,
exige una escalera de muchos cortes. Los **árboles y bosques oblicuos**, que cortan por combinaciones
lineales de variables, existen en la literatura general {cite:p}`murthy1994,breiman2001,menze2011`,
pero solo aparecieron aplicados en otros dominios, como el análisis de supervivencia en medicina
{cite:p}`jaeger2019`.

## Matriz de novedad

```{list-table} Búsquedas realizadas (bases de datos, cadenas y resultados).
:header-rows: 1
:name: tab-matriz-novedad

* - Base
  - Cadena
  - Fecha
  - Resultados
  - Relevantes
  - Observación
* - Web of Science
  - `TS=("default of credit card clients" OR "credit card default" OR "Taiwan credit" OR "Yeh and Lien" OR "UCI credit card")`
  - 29-oct-2026
  - 53
  - 53
  - Universo base; ningún modelo oblicuo
* - Web of Science
  - Bloque anterior AND `TS=("oblique decision tree" OR "oblique random forest" OR "oblique forest")`
  - 30-oct-2026
  - 0
  - 0
  - Ningún bosque oblicuo en este problema
* - Web of Science
  - Bloque anterior AND `TS=("rotation forest")`
  - 30-oct-2026
  - 0
  - 0
  - —
* - Web of Science
  - `TS=("credit scoring" OR "credit risk" OR "credit default" OR "loan default") AND TS=("rotation forest")`
  - 30-oct-2026
  - 9
  - 8
  - Trabajos cercanos en crédito, sin cortes oblicuos de Fisher
* - Scopus
  - Cuatro exportaciones sobre selección dinámica, Rotation Forest y bosques oblicuos en crédito
  - 30-oct-2026
  - 102
  - ≈10
  - El bosque oblicuo solo aparece en supervivencia médica
* - arXiv
  - `oblique forest credit` (resumen)
  - 11-nov-2026
  - 0
  - 0
  - —
* - arXiv
  - `oblique tree default` (resumen)
  - 11-nov-2026
  - 0
  - 0
  - —
* - arXiv
  - `Fisher split forest` (resumen)
  - 11-nov-2026
  - 1
  - 0
  - Modelo ecológico de Wright-Fisher, ajeno al problema
* - arXiv
  - `credit card default random forest` (resumen)
  - 11-nov-2026
  - 7
  - 0
  - Incumplimiento con LLM, GAN, circuitos cuánticos o bosques por ejes; ningún corte oblicuo. Tres usan este mismo conjunto de Taiwán
* - arXiv
  - `Taiwan credit card` (resumen)
  - 11-nov-2026
  - 2
  - 0
  - Redes bayesianas en el conjunto de Taiwán y una prueba de calibración; ningún corte oblicuo
```

## Diferenciación frente al trabajo más cercano

```{list-table} Métodos afines y en qué se diferencia CS-FOF.
:header-rows: 1
:name: tab-diferencias

* - Trabajo
  - Cómo elige la dirección del corte
  - Diferencia con CS-FOF
* - OC1 {cite:p}`murthy1994`
  - Búsqueda local aleatoria
  - Un solo árbol, sin costo
* - Forest-RC {cite:p}`breiman2001`
  - Combinaciones lineales aleatorias
  - Dirección no supervisada
* - Oblique RF {cite:p}`menze2011`
  - Discriminante lineal o ridge por nodo
  - Sin ponderación por costo ni competencia con cortes por ejes; no evaluado en crédito
* - ORSF {cite:p}`jaeger2019`
  - Cox penalizado por nodo
  - Supervivencia médica, otra tarea
* - PLTR {cite:p}`dumitrescu2022`
  - Reglas de árboles en una logística penalizada
  - Crédito, pero sin cortes oblicuos
* - GEV-GBDT {cite:p}`zhang2025`
  - Cortes por ejes con pérdida de valor extremo
  - Cambia la pérdida, no la geometría del corte
```

**Declaración de novedad.** Al mejor de nuestro conocimiento y según la búsqueda descrita en Web of
Science y Scopus (octubre de 2026) y arXiv (11 de noviembre de 2026), no se encontraron estudios que apliquen un bosque oblicuo con cortes
de Fisher ponderados por costo a la predicción del incumplimiento en tarjetas de crédito. El trabajo
más cercano en método es el bosque oblicuo de {cite:t}`menze2011`; CS-FOF se diferencia en que pondera
la dirección y el criterio de partición por el costo del incumplimiento y en que el corte oblicuo
compite con los cortes por ejes bajo ese mismo criterio.

```{note}
Pendiente antes de la versión final: completar la búsqueda por citas (*snowballing*) a partir de los
trabajos más cercanos, Menze et al. (2011) y Murthy et al. (1994).
```
