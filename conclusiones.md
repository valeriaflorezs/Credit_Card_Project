# Conclusiones y limitaciones

1. **Se propuso e implementó un modelo original**, CS-FOF, un bosque aleatorio con cortes oblicuos de
   Fisher ponderados por costo, programado desde cero con numpy, con tres propiedades demostradas y
   verificadas con pruebas unitarias.
2. **Es el mejor modelo del proyecto** en la validación anidada 5 × 5: F1 de 0.5479 ± 0.0054 y AUC-ROC
   de 0.7847, por encima de XGBoost (0.5466) y Random Forest (0.5452), con menor variabilidad.
3. **No es superior de forma significativa al mejor modelo de referencia.** Frente a XGBoost y Random
   Forest solo McNemar es significativa; la ventaja en F1 es pequeña. Sí supera con significancia y
   efecto grande a SVM, KNN, árbol de decisión, regresión logística y Naive Bayes.
4. **El corte oblicuo aporta y los cortes demasiado fuertes sobreajustan:** la ablación mostró que
   activar el corte oblicuo mejora el F1, y que más variables o más profundidad lo empeoran.
5. **El techo de este conjunto de datos es bajo:** los mejores modelos de la literatura y de este
   proyecto quedan cerca de F1 ≈ 0.55 y AUC ≈ 0.78, lo que limita cuánto puede separarse cualquier modelo.

## Limitaciones
- No se completaron, por tiempo de cómputo, el algoritmo genético y las técnicas `none`, `smote` y
  `adasyn` de CS-FOF, ni los conjuntos de datos adicionales y los estudios de robustez.
- El protocolo se documentó después de los experimentos (desviación declarada).
- Las probabilidades de CS-FOF están sesgadas hacia el incumplimiento por los pesos de costo y no se
  recalibraron.

## Trabajo futuro
Validación externa repetida para ganar potencia estadística, los tres conjuntos de datos adicionales,
calibración y predicción conforme, y una implementación compilada para reducir el tiempo de
entrenamiento.
