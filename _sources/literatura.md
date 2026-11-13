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
* - Google Scholar (citas hacia adelante)
  - Artículos que citan a {cite:t}`menze2011`, filtrados con `credit`
  - 12-nov-2026
  - 373 citas; 10 revisados (primera página del filtro)
  - 0
  - Incendios, antenas, cromatografía, mercados agrícolas; ningún bosque oblicuo en crédito
* - Google Scholar (citas hacia adelante)
  - Artículos que citan a {cite:t}`murthy1994`, filtrados con `credit`
  - 12-nov-2026
  - 1,475 citas; 17 revisados
  - 0
  - Árboles en general y optimización; destaca Bollwein y Westphal (2022), árbol oblicuo sin costo ni crédito
* - Google Scholar (citas hacia adelante)
  - Artículos que citan a {cite:t}`murthy1994`, filtrados con `"default" loan`
  - 12-nov-2026
  - 1,475 citas; 13 revisados
  - 0 en la aplicación; 1 cercano en método
  - **CSNL** {cite:p}`vadera2010`: árbol con nodos discriminantes sensibles al costo (se leyó el texto completo)
```

La búsqueda por citas (*snowballing*) se hizo hacia adelante desde los dos trabajos más cercanos en
método, Menze et al. (2011) y Murthy et al. (1994). No encontró ninguna aplicación de cortes oblicuos al
incumplimiento en tarjetas de crédito. Sí encontró el antecedente más cercano en método, CSNL
{cite:p}`vadera2010`, que se analiza en la siguiente sección. La búsqueda hacia atrás en las referencias de
Menze et al. no se completó porque no se consiguió el texto completo; se declara como limitación.

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
* - **CSNL** {cite:p}`vadera2010`
  - Discriminante **cuadrático** sensible al costo, sobre **2 variables** elegidas por ganancia de información individual
  - El más cercano. Es un solo árbol podado; MC+CSNL usa *bagging* solo para reetiquetar y entrega un árbol. El costo entra en el umbral de Bayes; los cortes por ejes solo se usan si la covarianza no se puede invertir, no compiten. Probado en 17 conjuntos, uno de crédito (*Credit Approval*), medido por costo y no por F1
* - Oblique DT con von Mises–Fisher {cite:p}`bollwein2022`
  - Optimización por entropía cruzada de la dirección (distribución de von Mises–Fisher, no discriminante de Fisher)
  - Un solo árbol, sin costo ni aplicación a crédito
* - Canonical correlation forests {cite:p}`rainforth2015`
  - Correlación canónica entre variables y clases por nodo
  - Bosque oblicuo sin costo ni competencia con cortes por ejes; no evaluado en crédito
* - Revisión de árboles sensibles al costo {cite:p}`lomax2013`
  - (revisión de métodos)
  - Ubica a CSNL dentro de los árboles sensibles al costo; ninguno es un bosque de cortes de Fisher ponderados
```

**Qué cambia CSNL en la declaración de novedad.** La idea de usar **análisis discriminante sensible al
costo** en los nodos de un árbol no es nueva: CSNL la propuso en 2010 y la probó, entre otros, en un
conjunto de aprobación de crédito (*Credit Approval*), casi el mismo que el conjunto *Australian* que
este proyecto usa como adicional. Lo que distingue a CS-FOF de CSNL es:

1. **Es un bosque aleatorio:** bootstrap, sorteo de variables por nodo y promedio de probabilidades. CSNL es
   un solo árbol.
2. **El corte es lineal (Fisher) sobre $q$ variables sorteadas**, con dispersión intra-clase ponderada por
   costo y regularizada. CSNL usa una frontera cuadrática sobre las dos mejores variables.
3. **El corte oblicuo compite con los cortes por ejes bajo el mismo Gini ponderado**, lo que da la
   Proposición 1 (nunca es peor que el mejor corte por ejes). En CSNL el corte discriminante siempre gana,
   salvo que la covarianza sea singular.
4. **El costo entra en los pesos**, tanto de la dirección como del criterio de partición. En CSNL entra en el
   umbral de la regla de Bayes.
5. **Se aplica al incumplimiento de pago de clientes de tarjeta**, con validación anidada, F1 y pruebas de
   hipótesis frente al estado del arte. CSNL se aplicó a la aprobación de solicitudes de crédito, que es otra
   tarea, con otra métrica.

**Declaración de novedad.** Al mejor de nuestro conocimiento, y según la búsqueda descrita en Web of
Science y Scopus (octubre de 2026), arXiv (11 de noviembre de 2026) y la búsqueda por citas en Google
Scholar (12 de noviembre de 2026), no se encontraron estudios que apliquen un **bosque aleatorio** cuyos
nodos usen cortes de **Fisher ponderados por costo en competencia con los cortes por ejes** a la
predicción del incumplimiento de pago en tarjetas de crédito.

Los trabajos más cercanos son:

- **Por la estructura de bosque oblicuo:** el bosque oblicuo de {cite:t}`menze2011`, que no pondera por costo
  ni hace competir los cortes.
- **Por los nodos discriminantes sensibles al costo:** CSNL {cite:p}`vadera2010`, que es un solo árbol con
  fronteras cuadráticas sobre dos variables y se evaluó en aprobación de crédito, no en incumplimiento de
  pago.

La contribución de CS-FOF no es la idea de un corte discriminante sensible al costo, que ya existía, sino
su combinación dentro de un bosque aleatorio con una regla de competencia que tiene una garantía demostrada
(Proposición 1).
