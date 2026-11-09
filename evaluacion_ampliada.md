# Evaluación ampliada: ablación, robustez y limitaciones

## Ablación y sensibilidad (pruebas de diseño)

Las variantes se evaluaron **solo con el entrenamiento del pliegue externo 1**, con validación interna
estratificada de 3 pliegues, sin usar ningún pliegue de prueba (`experiments/cs_fof_diseno/`).

```{list-table} Variantes de diseño de CS-FOF (F1 de la clase *default*, validación interna).
:header-rows: 1
:name: tab-ablacion

* - Variante
  - F1
  - Lectura
* - CS-FOF base (un candidato oblicuo, $q=\sqrt p$, $d=10$, $h=20$)
  - **0.5493**
  - Referencia
* - Sin corte oblicuo (ablación)
  - 0.5452
  - El corte oblicuo aporta +0.004
* - Oblicuo con $p/2$ variables
  - 0.5449
  - Un corte más fuerte sobreajusta
* - Dos candidatos oblicuos por nodo
  - 0.5482
  - Sin mejora
* - Más profundo ($d=14$, $h=10$)
  - 0.5238
  - Sobreajuste claro
* - 128 bins
  - 0.5458
  - Sin mejora
* - Random Forest / XGBoost (mejores hiperparámetros del pliegue 1)
  - 0.5450 / 0.5412
  - Referencias
```

Las variantes de regularización (profundidad 8 a 12, hoja mínima 20 a 40, $\varepsilon$ entre 0.01 y
0.5, `sqrt` o `log2`, 150 o 300 árboles) quedaron todas entre 0.546 y 0.550: el modelo **no depende de
un ajuste fino** de sus hiperparámetros, una señal de estabilidad frente al sobreajuste.

## Estabilidad entre pliegues

En la validación anidada 5 × 5, CS-FOF tiene la menor desviación estándar del F1 entre los cinco mejores
modelos (0.0054, frente a 0.0077–0.0091 de CatBoost, XGBoost, LightGBM y Random Forest), y sus cuatro
métodos de búsqueda dan resultados cercanos (0.5456 a 0.5479).

## Conjuntos de datos adicionales

Para comprobar la consistencia, los 10 modelos se evaluaron en tres conjuntos de crédito del repositorio
UCI con el mismo procedimiento: validación anidada 5 × 5, `class_weight` y búsqueda aleatoria de 25
configuraciones (`src/run_additional_datasets.py`).

```{list-table} F1 medio de la clase de riesgo en los 5 pliegues externos y rango medio en los 4 conjuntos.
:header-rows: 1
:name: tab-conjuntos

* - Modelo
  - Taiwán
  - German (1 000; 30 %)
  - Australian (690; 56 %)
  - Polish año 1 (7 027; 4 %)
  - Rango medio
* - CatBoost
  - **0.548**
  - **0.605**
  - 0.878
  - **0.733**
  - **1.25**
* - XGBoost
  - 0.547
  - 0.599
  - **0.880**
  - 0.682
  - 2.75
* - **CS-FOF**
  - **0.548**
  - 0.589
  - 0.874
  - 0.430
  - 4.75
* - Random Forest
  - 0.545
  - 0.600
  - 0.874
  - 0.446
  - 4.75
* - LightGBM
  - 0.546
  - 0.575
  - 0.874
  - 0.658
  - 4.75
* - SVM
  - 0.539
  - 0.601
  - 0.862
  - 0.167
  - 5.75
* - KNN
  - 0.526
  - 0.560
  - 0.877
  - 0.138
  - 6.75
* - Regresión logística
  - 0.518
  - 0.600
  - 0.872
  - 0.130
  - 7.00
* - Árbol de decisión
  - 0.525
  - 0.521
  - 0.860
  - 0.429
  - 8.25
* - Naive Bayes
  - 0.499
  - 0.591
  - 0.837
  - 0.074
  - 9.00
```

La prueba de Friedman con la corrección de Iman–Davenport rechaza la igualdad de los 10 modelos en los
4 conjuntos ($F=4.38$, $p=0.0013$), con CatBoost en el primer lugar. Ninguna comparación de CS-FOF con
Wilcoxon y Holm es significativa (con 4 conjuntos el p mínimo es 0.125). Lectura por conjunto:

- **German:** CS-FOF tiene el mayor AUC-ROC de los 10 (0.790) y un AUC-PR de 0.627, pero su F1 a umbral
  0.5 queda en el puesto 7 (0.589).
- **Australian:** todos los modelos basados en árboles quedan dentro de 0.006 en F1; CS-FOF es cuarto.
- **Polish:** con 4 % de quiebras y razones financieras de colas muy pesadas, los tres métodos de
  *boosting* superan con amplitud a los dos bosques (CatBoost 0.733 frente a CS-FOF 0.430 y Random
  Forest 0.446). El *boosting* corrige iterativamente los casos raros mal clasificados, mientras que el
  promedio de árboles independientes los diluye; la ponderación por costo de CS-FOF no basta para
  compensarlo.

**CS-FOF no es consistentemente superior en los conjuntos adicionales.** Su desempeño es equivalente al
de Random Forest, el modelo del que parte, y queda por detrás de CatBoost y XGBoost cuando el
desbalance es extremo.

## Robustez

Estudios **exploratorios** (no estaban en el protocolo original) sobre los cinco mejores modelos en
Taiwán (`experiments/robustez/robustez.py`). Para no hacer una búsqueda nueva, cada modelo usa los
hiperparámetros que eligió en cada pliegue externo; el ruido y el submuestreo se aplican solo al
entrenamiento, y la prueba siempre es limpia y completa. Son 5 pliegues por nivel, sin pruebas de
hipótesis, así que las diferencias pequeñas deben leerse con cautela.

**Ruido en las etiquetas.** Se invierte al azar una fracción de las etiquetas de entrenamiento (la misma
para todos los modelos).

```{list-table} F1 medio en 5 pliegues según el porcentaje de etiquetas de entrenamiento invertidas.
:header-rows: 1
:name: tab-ruido

* - Modelo
  - 0 %
  - 5 %
  - 10 %
  - 20 %
* - **CS-FOF**
  - **0.5483**
  - 0.5455
  - **0.5455**
  - **0.5336**
* - CatBoost
  - 0.5471
  - **0.5469**
  - 0.5416
  - 0.5301
* - XGBoost
  - 0.5448
  - 0.5430
  - 0.5401
  - 0.5252
* - LightGBM
  - 0.5459
  - 0.5443
  - 0.5408
  - 0.5304
* - Random Forest
  - 0.5434
  - 0.5423
  - 0.5414
  - 0.5303
```

Todos pierden entre 0.013 y 0.020 de F1 con 20 % de ruido. CS-FOF tiene el mayor F1 en tres de los cuatro
niveles y pierde menos que los tres métodos de *boosting* (−0.015 frente a −0.016 a −0.020); Random Forest
es el que menos pierde (−0.013).

**Tamaño de la muestra.** Se entrena con una fracción estratificada del entrenamiento de cada pliegue.

```{list-table} F1 medio en 5 pliegues según la fracción del entrenamiento usada.
:header-rows: 1
:name: tab-tamano

* - Modelo
  - 10 %
  - 25 %
  - 50 %
  - 100 %
* - **CS-FOF**
  - 0.5356
  - **0.5429**
  - **0.5473**
  - **0.5483**
* - CatBoost
  - 0.5290
  - 0.5379
  - 0.5442
  - 0.5471
* - XGBoost
  - 0.5244
  - 0.5370
  - 0.5392
  - 0.5448
* - LightGBM
  - 0.5363
  - 0.5415
  - 0.5437
  - 0.5459
* - Random Forest
  - **0.5375**
  - 0.5415
  - 0.5405
  - 0.5434
```

Con 25 % o más de los datos CS-FOF queda primero; con 10 % (unos 2 400 clientes) Random Forest y LightGBM
quedan ligeramente por encima. Los métodos de *boosting* son los que más pierden con pocos datos (XGBoost
−0.020 y CatBoost −0.018 entre 100 % y 10 %, frente a −0.013 de CS-FOF). Hay un sesgo a favor de todos los
modelos por igual: sus hiperparámetros se eligieron con el entrenamiento completo.

**Calibración.** Con los pesos de clase, todos los modelos sobreestiman la probabilidad de
incumplimiento (probabilidad media de 0.40 a 0.44 frente a una tasa real de 0.22).

```{list-table} Calibración de las probabilidades fuera de pliegue (29 965 clientes; tasa real 0.221).
:header-rows: 1
:name: tab-calibracion

* - Modelo
  - Brier
  - Log-loss
  - ECE
  - Probabilidad media
* - **CS-FOF**
  - 0.1679
  - 0.5177
  - 0.1797
  - 0.401
* - Random Forest
  - 0.1725
  - 0.5276
  - 0.1891
  - 0.410
* - XGBoost
  - 0.1750
  - 0.5347
  - 0.1962
  - 0.417
* - CatBoost
  - 0.1785
  - 0.5435
  - 0.2042
  - 0.425
* - LightGBM
  - 0.1899
  - 0.5674
  - 0.2197
  - 0.441
```

CS-FOF tiene las probabilidades mejor calibradas de los cinco en las tres métricas, aunque ninguno está
bien calibrado; para usar las probabilidades como riesgo real hace falta recalibrarlas (por ejemplo, con
regresión isotónica), lo que queda como trabajo futuro.

**Costo.** CS-FOF es el más lento: unos 13 s por ajuste con 4 procesos, frente a 1–3 s de los demás.

## Lo que no se ejecutó

Por tiempo de cómputo no se ejecutaron: la validación externa repetida (no cambiaría el veredicto, ver
`experiments/protocolo.md`), el estudio de cambio de distribución, la recalibración, la predicción
conforme y la ablación con validación anidada 5 × 5. Se declaran como desviaciones y como trabajo futuro.
