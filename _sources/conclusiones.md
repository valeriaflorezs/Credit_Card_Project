# Conclusiones y limitaciones

1. **Se propuso e implementó un modelo original**, CS-FOF, un bosque aleatorio con cortes oblicuos de
   Fisher ponderados por costo, programado desde cero con numpy, con tres propiedades demostradas y
   verificadas con pruebas unitarias.
2. **Obtiene el mayor F1 medio, sin diferencia significativa.** En la validación anidada
   5 × 5 obtiene un F1 de 0.5479 ± 0.0054, por encima de XGBoost (0.5466) y de Random Forest (0.5452), con el
   mayor AUC-ROC (0.7847) de los 8 modelos y la menor variabilidad entre pliegues de los tres mejores.
3. **No es superior de forma significativa a los mejores modelos de referencia.** Frente a XGBoost y Random
   Forest, ni Nadeau–Bengio, ni Wilcoxon, ni DeLong son significativas. Sí supera con
   significancia y efecto grande a SVM, KNN, árbol de decisión, regresión logística y Naive Bayes.
   Con el umbral elegido dentro del entrenamiento de cada modelo, el empate se mantiene: CS-FOF queda
   0.0007 por delante de XGBoost, sin significancia.
4. **No es consistente en otros conjuntos de crédito.** En German y Australian queda en el grupo de
   cabeza (con el mayor AUC-ROC en German), pero en Polish, con 4 % de quiebras, XGBoost
   lo supera con amplitud. En el rango medio de los 4 conjuntos, XGBoost es primero (2.0), Random Forest
   segundo (3.0) y CS-FOF tercero (3.25).
5. **Mantiene el mayor F1 en los estudios de robustez en Taiwán** (exploratorios): tiene el mayor F1 con 10 % y
   20 % de etiquetas ruidosas, con 25 % o más de los datos de entrenamiento y al entrenar con un grupo de
   edad o de educación y probar con otro. Sus probabilidades sin recalibrar son las mejor calibradas de los
   tres mejores modelos, aunque tras la regresión isotónica todos quedan iguales. Su predicción conforme
   cubre el 90 % nominal en ambas clases.
6. **El corte oblicuo aporta y los cortes demasiado fuertes sobreajustan:** la ablación mostró que
   activar el corte oblicuo mejora el F1, y que más variables o más profundidad lo empeoran.
7. **Los mejores modelos quedan muy cerca entre sí:** los de la literatura y los de este proyecto rondan
   F1 ≈ 0.55 y AUC ≈ 0.78 con estas variables. Esto no prueba que exista un techo; solo indica que, con la
   información disponible, las diferencias entre buenos modelos son pequeñas y cuesta separarlas
   estadísticamente.
8. **Un ensamble a partir de CS-FOF sube el F1, poco.** Para mejorar el F1 se hizo un ensamble (*stacking*) con CS-FOF, XGBoost,
   CatBoost, LightGBM, Random Forest, Extra Trees, HistGradientBoosting y regresión logística: HIB (capítulo «Del CS-FOF al HIB»).
   Con semillas nuevas (25 pliegues) obtiene F1 0.5505 frente a 0.5468 de XGBoost (+0.0037) y supera a los 7 modelos base con
   Wilcoxon con Holm, DeLong y McNemar. La ventaja es pequeña: Nadeau–Bengio no rechaza frente a XGBoost (p = 0.049 sin ajustar) y un promedio
   simple de los mismos 8 modelos logra casi lo mismo (+0.0004, p = 0.33), así que la ventaja viene de combinar modelos y no de la regla de pesos.
   Solo se evaluó en Taiwán.

Según la sección 6.4 del Entregable 3, la condición R3 (superioridad estadística y consistencia) **no se
cumple**. Se reporta así, sin ajustar el modelo después de ver los resultados de prueba.

## Limitaciones
- El protocolo se documentó después de los experimentos (desviación declarada en
  `experiments/protocolo.md`).
- CS-FOF solo se evaluó con pesos de clase.
- Las 112 combinaciones comparan los siete modelos de referencia exigidos por el curso. CatBoost y LightGBM entran solo como
  referencias y como miembros del ensamble HIB, sin el mismo presupuesto de ajuste.
- HIB se evaluó solo en Taiwán, con particiones distintas en cada corrida, y su ventaja sobre XGBoost no sobrevive a la t corregida de
  Nadeau–Bengio; el protocolo se editó después para quitar los conjuntos adicionales y el *tag* de Git solo existe en local.
- Con 5 pliegues externos y 4 conjuntos, las pruebas pareadas tienen poca potencia; no se ejecutó la
  validación externa repetida.
- Los estudios de robustez son exploratorios: usan los hiperparámetros elegidos con los datos completos y
  no tienen pruebas de hipótesis.
- Las probabilidades de todos los modelos están sesgadas hacia el incumplimiento por los pesos de costo.
  La regresión isotónica lo corrige (ECE de 0.18 a 0.003), pero entonces la ventaja de calibración de
  CS-FOF desaparece.
- CS-FOF es unas 12 veces más lento de entrenar que XGBoost y unas 50 veces más lento al predecir
  (253 ms por cada 1,000 clientes), porque está escrito en numpy sin código compilado.
- CS-FOF falla más que XGBoost justo en el grupo más grande de la cartera: los clientes al día en
  septiembre (77 % de los clientes, F1 0.261 frente a 0.289).
- Con 90 % de confianza, la predicción conforme deja a la mitad de los clientes en el conjunto
  {pago, impago}: con estas variables, una parte grande de los clientes sigue siendo difícil de clasificar.
- El cambio de distribución solo se simuló por grupos de edad y educación, porque los datos no tienen
  fechas; no reemplaza una validación con datos de otro periodo o de otro banco.

## Trabajo futuro
Una versión con *boosting* de los cortes oblicuos de Fisher para los casos de desbalance extremo,
validación externa repetida para ganar potencia estadística, una dirección de corte no lineal (o varias
direcciones por nodo) para los clientes al día, donde CS-FOF pierde frente a XGBoost, y una
implementación compilada para reducir el tiempo de entrenamiento y de predicción.
