# Conclusiones y limitaciones

1. **Se propuso e implementó un modelo original**, CS-FOF, un bosque aleatorio con cortes oblicuos de
   Fisher ponderados por costo, programado desde cero con numpy, con tres propiedades demostradas y
   verificadas con pruebas unitarias.
2. **Empata con el mejor método del estado del arte en el problema principal.** En la validación anidada
   5 × 5 obtiene un F1 de 0.5479 ± 0.0054, a 0.0001 de CatBoost (0.5480), con el mayor AUC-ROC (0.7847)
   y la menor variabilidad entre pliegues de los 10 modelos.
3. **No es superior de forma significativa al mejor modelo de referencia.** Frente a CatBoost, XGBoost,
   LightGBM y Random Forest, ni Nadeau–Bengio, ni Wilcoxon, ni DeLong son significativas. Sí supera con
   significancia y efecto grande a SVM, árbol de decisión, regresión logística y Naive Bayes.
4. **No es consistente en otros conjuntos de crédito.** En German y Australian queda en el grupo de
   cabeza (con el mayor AUC-ROC en German), pero en Polish, con 4 % de quiebras, los métodos de *boosting*
   lo superan con amplitud. En el rango medio de los 4 conjuntos, CatBoost es primero (1.25) y CS-FOF
   empata en el tercer lugar con Random Forest y LightGBM (4.75).
5. **Es más robusto que el *boosting* en Taiwán** (estudios exploratorios): tiene el mayor F1 con 10 % y
   20 % de etiquetas ruidosas, con 25 % o más de los datos de entrenamiento, y las probabilidades mejor
   calibradas de los cinco mejores modelos.
6. **El corte oblicuo aporta y los cortes demasiado fuertes sobreajustan:** la ablación mostró que
   activar el corte oblicuo mejora el F1, y que más variables o más profundidad lo empeoran.
7. **El techo de este conjunto de datos es bajo:** los mejores modelos de la literatura y de este
   proyecto quedan cerca de F1 ≈ 0.55 y AUC ≈ 0.78, lo que limita cuánto puede separarse cualquier modelo.

Según la sección 6.4 del Entregable 3, la condición R3 (superioridad estadística y consistencia) **no se
cumple**. Se reporta así, sin ajustar el modelo después de ver los resultados de prueba.

## Limitaciones
- El protocolo se documentó después de los experimentos (desviación declarada en
  `experiments/protocolo.md`).
- CS-FOF solo se evaluó con pesos de clase. LightGBM y CatBoost completaron sus 16 combinaciones; en ambos,
  pesos de clase fue la mejor técnica.
- Con 5 pliegues externos y 4 conjuntos, las pruebas pareadas tienen poca potencia; no se ejecutó la
  validación externa repetida.
- Los estudios de robustez son exploratorios: usan los hiperparámetros elegidos con los datos completos y
  no tienen pruebas de hipótesis.
- Las probabilidades de todos los modelos están sesgadas hacia el incumplimiento por los pesos de costo
  y no se recalibraron.
- CS-FOF es entre 4 y 15 veces más lento de entrenar que los demás bosques y métodos de *boosting*.

## Trabajo futuro
Una versión con *boosting* de los cortes oblicuos de Fisher para los casos de desbalance extremo,
validación externa repetida para ganar potencia estadística, recalibración y predicción conforme, y una
implementación compilada para reducir el tiempo de entrenamiento.
