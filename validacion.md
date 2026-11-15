# Validación estadística del modelo original

## Protocolo

Todos los modelos se evalúan con la misma validación cruzada anidada estratificada (5 pliegues externos ×
5 internos, semilla 42), el mismo preprocesamiento y el mismo presupuesto de búsqueda
(`experiments/protocolo.md`). La métrica principal es el F1 de la clase *default*. Las desviaciones
estándar usan `ddof=1`. Se compara CS-FOF con los 7 modelos de referencia exigidos por el curso: regresión logística, KNN, Naive
Bayes, SVM, árbol de decisión, Random Forest y XGBoost. Las cifras se reproducen con
`experiments/paper/resultados_paper.py`, `experiments/comparacion_final/pruebas.py` y
`experiments/comparacion_final/delong.py`.

## Resultados

```{list-table} Mejor combinación de cada modelo en los 5 pliegues externos.
:header-rows: 1
:name: tab-ranking

* - Modelo
  - Técnica
  - Búsqueda
  - F1 (media ± d.e.)
  - AUC-ROC
* - **CS-FOF**
  - class_weight
  - aleatoria
  - **0.5479 ± 0.0054**
  - **0.7847**
* - XGBoost
  - class_weight
  - Optuna
  - 0.5466 ± 0.0091
  - 0.7828
* - Random Forest
  - class_weight
  - aleatoria
  - 0.5452 ± 0.0077
  - 0.7831
* - SVM
  - SMOTE
  - aleatoria
  - 0.5387 ± 0.0048
  - 0.7638
* - KNN
  - class_weight
  - aleatoria
  - 0.5264 ± 0.0122
  - 0.7629
* - Árbol de decisión
  - class_weight
  - Optuna
  - 0.5254 ± 0.0043
  - 0.7569
* - Regresión logística
  - class_weight
  - genético
  - 0.5182 ± 0.0104
  - 0.7449
* - Naive Bayes
  - SMOTE
  - malla
  - 0.4987 ± 0.0078
  - 0.7454
```

CS-FOF obtiene el mayor F1 medio (0.5479 frente a 0.5466 de XGBoost), el mayor AUC-ROC y la menor
variabilidad entre los tres mejores modelos. Ninguna de esas diferencias es significativa. Sus cuatro métodos
de búsqueda dan resultados cercanos: aleatoria 0.5479, genético 0.5476, malla 0.5468 y Optuna 0.5456.

## Pruebas de hipótesis

**Friedman y Nemenyi.** Sobre la mejor combinación de los 8 modelos, la prueba de Friedman rechaza la
igualdad de desempeño ($\chi^2=32.93$, $p=2.7\times10^{-5}$). Los mejores rangos medios son CS-FOF (1.4),
XGBoost (2.2) y Random Forest (2.4). Con 5 pliegues y 8 modelos la diferencia crítica de Nemenyi es 4.70,
así que la prueba solo separa a los mejores de los modelos clásicos más débiles.

```{figure} results/figures/cd_diagram_final.png
:name: fig-cd-final
:width: 100%

Diagrama de diferencia crítica de Nemenyi con CS-FOF incluido. Cada modelo se ubica en su rango medio
(1 es el mejor) sobre los 5 pliegues externos; las barras azules unen modelos cuya diferencia de rangos es
menor que la diferencia crítica (4.70) y que, por tanto, no se distinguen. CS-FOF (1.4) queda en el
mismo grupo que XGBoost (2.2) y Random Forest (2.4).
```

```{list-table} CS-FOF frente a la mejor combinación de cada modelo. p-valores ajustados con Holm (7 comparaciones).
:header-rows: 1
:name: tab-pruebas

* - Rival
  - Diferencia de F1
  - Gana en
  - p Nadeau–Bengio (Holm)
  - p Wilcoxon (Holm)
  - δ de Cliff
* - XGBoost
  - +0.0014
  - 4 de 5
  - 0.627
  - 0.438
  - 0.36
* - Random Forest
  - +0.0027
  - 4 de 5
  - 0.619
  - 0.438
  - 0.36
* - SVM
  - +0.0092
  - 5 de 5
  - **0.010**
  - 0.438
  - 0.76
* - KNN
  - +0.0216
  - 5 de 5
  - **0.034**
  - 0.438
  - 0.92
* - Árbol de decisión
  - +0.0225
  - 5 de 5
  - **0.003**
  - 0.438
  - 1.00
* - Regresión logística
  - +0.0297
  - 5 de 5
  - **0.010**
  - 0.438
  - 1.00
* - Naive Bayes
  - +0.0493
  - 5 de 5
  - **< 0.001**
  - 0.438
  - 1.00
```

Con 5 pliegues, la prueba de Wilcoxon bilateral no puede bajar de $p=0.0625$, por lo que nunca rechaza.

**DeLong y McNemar.** Se reajustaron los modelos con los hiperparámetros elegidos en cada pliegue para
obtener predicciones fuera de pliegue de los 29 965 clientes.

```{list-table} DeLong (AUC) y McNemar (errores a umbral 0.5) sobre predicciones fuera de pliegue.
:header-rows: 1
:name: tab-delong

* - Rival
  - AUC CS-FOF / rival
  - p DeLong
  - Solo falla CS-FOF / solo falla el rival
  - p McNemar
* - XGBoost
  - 0.7833 / 0.7827
  - 0.561
  - 434 / 827
  - < 0.001
* - Random Forest
  - 0.7833 / 0.7822
  - 0.171
  - 313 / 615
  - < 0.001
```

McNemar es significativa contra todos porque CS-FOF comete menos errores exclusivos, pero esto refleja
sobre todo que sus probabilidades están mejor calibradas alrededor del umbral 0.5 (capítulo de
evaluación ampliada), no un mejor ordenamiento de los clientes.

**Intervalos de confianza bootstrap de la diferencia de F1.** Las secciones 6.5 y 6.7 de la guía piden un
intervalo para la diferencia de desempeño. Se construyó con las mismas predicciones fuera de pliegue de los
29 965 clientes y umbral 0.5 (`experiments/comparacion_final/bootstrap_f1.py`):

- **Remuestreo pareado:** se toman clientes con reemplazo, estratificando por clase, y en cada réplica se
  calcula F1(CS-FOF) − F1(rival) sobre los mismos clientes. Son 10 000 réplicas con semilla 42.
- **Intervalos:** se reportan el percentil y el BCa. La aceleración del BCa sale de un jackknife exacto: como
  el F1 depende solo de los conteos de la matriz de confusión, basta agrupar a los clientes en sus 16 tipos de
  aciertos y errores.
- **Valor p:** se calcula a partir de las réplicas y se corrige con Holm sobre los 2 rivales.

```{list-table} Diferencia de F1 (CS-FOF menos rival) con predicciones fuera de pliegue, umbral 0.5. F1 de CS-FOF: 0.5484.
:header-rows: 1
:name: tab-bootstrap

* - Rival
  - Diferencia
  - IC 95 % percentil
  - IC 95 % BCa
  - Réplicas en que gana CS-FOF
  - p bootstrap
  - p Holm
* - XGBoost
  - +0.0035
  - [−0.0006, +0.0077]
  - [−0.0006, +0.0076]
  - 96 %
  - 0.087
  - 0.087
* - Random Forest
  - +0.0050
  - [+0.0015, +0.0085]
  - [+0.0015, +0.0085]
  - 99.7 %
  - 0.005
  - **0.011**
```

**Lectura de los intervalos:**

- **XGBoost:** el intervalo contiene el cero. Frente a XGBoost, que es el mejor modelo de referencia, el rango
  de diferencias compatibles va de −0.001 a +0.008.
- **Random Forest:** la diferencia sí es significativa tras Holm, pero son 0.5 puntos de F1. Es una ventaja
  pequeña en la práctica. Además no es una ventaja en todo: CS-FOF detecta 102 incumplidores menos que Random
  Forest (3,893 frente a 3,995), pero da 404 falsas alarmas menos (3,675 frente a 4,079). El F1 mejora porque
  el segundo efecto pesa más.
- **Alcance:** este bootstrap es **condicional a los modelos ya entrenados**. Mide cuánto cambia la diferencia
  si los clientes fueran otros, pero no la variación por volver a entrenar. Esa variación la recoge la t
  corregida de Nadeau–Bengio, que frente a Random Forest no es significativa (p Holm = 0.62).
- **Las dos pruebas responden preguntas distintas:** «con estos modelos, ¿CS-FOF clasifica mejor a esta
  población?» frente a «¿el método CS-FOF, reentrenado, supera al método Random Forest?». La segunda es la
  que pide la condición R3.

**Umbral elegido dentro del entrenamiento (exploratorio).** Todas las comparaciones anteriores usan el umbral
0.5. Para comprobar que el resultado no depende de esa elección, cada uno de los tres mejores modelos recibió su
propio umbral, con el mismo protocolo (`experiments/comparacion_final/umbral_interno.py`):

1. **Predicciones internas.** En cada pliegue externo se usa solo su entrenamiento. Con los hiperparámetros
   elegidos ahí se generan predicciones internas fuera de pliegue (5 pliegues).
2. **Elección del umbral.** Con esas predicciones internas se elige el umbral que maximiza el F1.
3. **Aplicación.** El umbral queda congelado y se aplica a la prueba externa, que no intervino en ninguna
   elección.

Es exploratorio porque se decidió después de ver los resultados.

```{list-table} F1 en la prueba externa con umbral 0.5 y con el umbral elegido en el entrenamiento (media de 5 pliegues).
:header-rows: 1
:name: tab-umbral-interno

* - Modelo
  - Umbral elegido (rango)
  - F1 con 0.5
  - F1 con umbral ajustado
  - Cambio
* - CS-FOF
  - 0.51 (0.50 a 0.53)
  - 0.5483
  - 0.5481
  - −0.0003
* - XGBoost
  - 0.53 (0.51 a 0.56)
  - 0.5448
  - 0.5474
  - +0.0026
* - Random Forest
  - 0.53 (0.52 a 0.54)
  - 0.5434
  - 0.5452
  - +0.0019
```

**Qué cambia al ajustar el umbral:**

- **XGBoost y Random Forest ganan entre 0.002 y 0.003 de F1;** CS-FOF no gana nada. Su umbral
  óptimo ya estaba en 0.51, porque sus probabilidades están mejor centradas, como se vio en la calibración.
- **Con umbrales ajustados, CS-FOF queda 0.0007 por delante de XGBoost** (0.5481 frente a 0.5474). No es
  significativo:
  - Nadeau–Bengio da p = 0.86 y Wilcoxon p = 0.63, ambos sin corregir.
  - El bootstrap pareado del F1 global da un IC 95 % de [−0.0033, +0.0046].
- **La diferencia frente a Random Forest que el bootstrap mostraba con 0.5 deja de ser significativa:** queda
  en +0.0028, con IC 95 % de [−0.0005, +0.0061].
- **Lectura:** parte de la pequeña ventaja de CS-FOF con el umbral 0.5 venía de que sus probabilidades
  quedan mejor centradas en ese corte, no de que separe mejor a los clientes. Cuando cada modelo usa su propio
  umbral, los tres quedan empatados.

**Qué mide cada prueba.** Las pruebas de este capítulo no responden la misma pregunta, y por eso no se
cuentan como tres demostraciones de una mejora en F1:

```{list-table} Pregunta que contesta cada prueba y su resultado frente a XGBoost, el mejor modelo de referencia.
:header-rows: 1
:name: tab-que-mide

* - Prueba
  - Métrica que compara
  - Fuente de variación que considera
  - Frente a XGBoost
* - Friedman con Nemenyi
  - Rango del F1 entre los 8 modelos
  - Pliegues externos
  - Mismo grupo (diferencia de rangos 0.8 < 4.70)
* - t corregida de Nadeau–Bengio (Holm)
  - F1 por pliegue
  - Pliegues, con corrección por solapamiento del entrenamiento
  - No significativa (p = 0.63)
* - Wilcoxon de rangos con signo (Holm)
  - F1 por pliegue
  - Pliegues
  - No significativa (con 5 pliegues, p ≥ 0.0625)
* - Bootstrap pareado (BCa)
  - F1 fuera de pliegue
  - Clientes, con los modelos fijos
  - No significativa (IC 95 % [−0.001, +0.008])
* - DeLong
  - AUC-ROC, no F1
  - Clientes, con los modelos fijos
  - No significativa (p = 0.561)
* - McNemar
  - Errores a umbral 0.5, no F1
  - Clientes, con los modelos fijos
  - Significativa: CS-FOF tiene menos errores exclusivos, sobre todo por su calibración alrededor de 0.5
```

De las cuatro pruebas sobre el F1, ninguna muestra superioridad frente a XGBoost. DeLong y McNemar se
reportan como información complementaria: no evalúan la métrica principal.

## Veredicto de superioridad

Según la definición operativa del Entregable 3 (sección 6.4), **CS-FOF no es superior de forma
estadísticamente significativa a los mejores modelos de referencia** (XGBoost y Random Forest): frente a los
dos, Nadeau–Bengio y Wilcoxon no son significativas, el intervalo bootstrap de la diferencia de F1 frente a
XGBoost contiene el cero ([−0.001, +0.008]) y el tamaño del efecto es medio (δ de Cliff 0.36). El único
intervalo que excluye el cero es frente a Random Forest (+0.005 de F1), pero está condicionado a los modelos
ya entrenados y la prueba que considera el reentrenamiento (Nadeau–Bengio) no lo confirma. Con el umbral
elegido dentro del entrenamiento de cada modelo, esa diferencia también desaparece y los tres quedan
empatados. En los tres conjuntos de datos adicionales tampoco es consistente (capítulo de evaluación
ampliada). Sí es significativamente mejor que SVM, KNN, árbol de decisión, regresión logística y Naive Bayes
con tamaños de efecto grandes, y obtiene el mayor F1 medio en el problema principal, con la menor
variabilidad entre pliegues de los tres mejores.
