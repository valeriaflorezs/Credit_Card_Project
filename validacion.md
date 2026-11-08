# Validación estadística del modelo original

## Protocolo

Todos los modelos se evalúan con la misma validación cruzada anidada estratificada (5 pliegues externos ×
5 internos, semilla 42), el mismo preprocesamiento y el mismo presupuesto de búsqueda
(`experiments/protocolo.md`). La métrica principal es el F1 de la clase *default*. Las desviaciones
estándar usan `ddof=1`. Las cifras de esta sección se reproducen con
`experiments/comparacion_final/pruebas.py` y `experiments/comparacion_final/delong.py`.

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

```{note}
LightGBM y CatBoost (métodos recientes del estado del arte) se agregan a esta tabla cuando termine su
corrida (`src/run_boosting_recientes_5x5.py`).
```

CS-FOF obtiene el mayor F1 medio y el mayor AUC-ROC, con la menor variabilidad entre los tres mejores
modelos. Las otras dos combinaciones de CS-FOF también quedan cerca: malla 0.5468 ± 0.0088 y Optuna
0.5456 ± 0.0079.

## Pruebas de hipótesis

**Friedman y Nemenyi.** Sobre la mejor combinación de los 8 modelos, la prueba de Friedman rechaza la
igualdad de desempeño ($\chi^2=32.93$, $p=2.7\times10^{-5}$). CS-FOF tiene el mejor rango medio (1.4),
seguido de XGBoost (2.2) y Random Forest (2.4). Con 5 pliegues la diferencia crítica de Nemenyi es
4.70, de modo que la prueba solo separa a CS-FOF de la regresión logística y de Naive Bayes.

```{list-table} CS-FOF frente a la mejor combinación de cada modelo. p-valores ajustados con Holm.
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

## Veredicto de superioridad

Según la definición operativa del Entregable 3 (sección 6.4), **CS-FOF no es superior de forma
estadísticamente significativa al mejor modelo de referencia**. Su F1 medio es el mayor, pero frente a
XGBoost y Random Forest solo la prueba de McNemar es significativa; Nadeau–Bengio, Wilcoxon y DeLong no
lo son, y el tamaño del efecto es pequeño (δ = 0.36). Además, no se ejecutaron los conjuntos de datos
adicionales. Sí es significativamente mejor que SVM, KNN, árbol de decisión, regresión logística y
Naive Bayes, con tamaños de efecto grandes. En términos prácticos, CS-FOF comete casi la mitad de errores
exclusivos que XGBoost (434 frente a 827) con una variabilidad entre pliegues menor.
