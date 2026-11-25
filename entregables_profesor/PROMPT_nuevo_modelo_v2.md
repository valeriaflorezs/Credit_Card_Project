# Prompt v2: diseñar un componente NUEVO y complementario para superar a XGBoost y Random Forest en default de tarjetas de crédito

## Rol y objetivo
Eres un investigador de machine learning tabular. Necesitamos un modelo (o regla de combinación) NO neuronal, programado desde cero en
numpy/scipy y compatible con scikit-learn, que mejore de forma ESTADÍSTICAMENTE SIGNIFICATIVA el F1 de la clase default (UCI Taiwán, 29 965 filas,
22.1 % de morosos, datos de 2005) frente a XGBoost y Random Forest, además de SVM, KNN, regresión logística, árbol de decisión y Naive Bayes.
No es necesario que sea sensible al costo. Puede ser un modelo nuevo, un componente que aporte información distinta a los árboles de gradiente,
o una regla de combinación nueva. Si la evidencia dice que es imposible con este dato, dilo con números y propón el cierre honesto.
No prometas significancia: demuestra, con un piloto barato, si es alcanzable.

## Qué cambió respecto a la v1 (evidencia nueva, ya medida)
1. **CS-FOF no aporta al ensamble.** En la corrida confirmatoria (semillas 5000-5004, 25 pliegues) el ensamble HIB (8 miembros, pesos por F1 suave)
   dio F1 0.5502 y HIB sin CS-FOF 0.5505 (p Nadeau-Bengio = 0.72). Contra XGBoost +0.0039 y contra RF +0.0031: gana en 19 de 25 pliegues y Cliff
   0.20 y 0.17, pero p Nadeau-Bengio = 0.17 y 0.16 (Holm 0.32), Nemenyi 0.86 y 0.97. Solo Wilcoxon-Holm (0.0016) es significativo. No sirve para R3.
2. **Los modelos de árboles son casi redundantes.** Correlación de Spearman entre sus puntajes en prueba: 0.93 a 0.99 (XGB-LGBM 0.983, XGB-RF 0.982,
   CatBoost-XGB 0.957, ExtraTrees-XGB_d4 0.931). Por eso combinarlos da como máximo +0.001 de AUC sobre el mejor miembro. Los mejores ensambles
   (F1 ≈ 0.550) NO superan a CatBoost solo (0.550): la "mejora" frente a XGBoost viene de CatBoost, no de la combinación.
3. **Lo único realmente distinto es la regresión logística** (correlación 0.75-0.80 con los árboles), pero es débil (AUC 0.744, F1 0.523). Un miembro
   complementario útil debería tener correlación < 0.90 con los GBDT Y AUC ≥ 0.78. Ese es el hueco a llenar.
4. **Dónde falla el F1:** clientes al día (PAY_0 ≤ 0, 77 % de la cartera): AUC 0.69 y F1 0.25; clientes con atraso (PAY_0 > 0): AUC 0.74 y F1 0.71.
   Cualquier ganancia debe venir de los al día o de mejor calibración entre regímenes.
5. **Techo de información:** todos los modelos estables dan AUC ≈ 0.785 y F1 ≈ 0.546-0.550. Aumentar repeticiones NO ayuda a Nadeau-Bengio (la varianza se
   acota por 1/J + n_test/n_train ≥ 0.25). Con 25 pliegues, para que Holm sobre 7 contrastes dé p < 0.05 hace falta una ganancia media de F1 de
   aproximadamente **+0.0065 a +0.0075** sobre XGBoost y RF (hoy máximo +0.004). Es casi el doble de lo logrado.
6. Descartado (no repetir): stacking LR, FWLS por PAY_0, Caruana, promedios (2 582 combinaciones), WoE en PAY_x, variables derivadas de PAY, variables
   de razones de facturas/pagos (AUC +0.002-0.004), linear_tree, Rotation Forest propio, calibración, costo/class_weight (equivale a mover el umbral),
   corte oblicuo (+0.002). Los PAY_x son numéricos en apariencia pero categóricos en naturaleza (riesgo no monótono); los árboles ya lo manejan y solo la
   regresión logística mejora con codificación (+0.023 AUC).

## Reglas de la rúbrica (Entregable 3)
- R1 Novedad para ESTE problema (default de tarjetas): búsqueda documentada en ≥ 3 bases, ≥ 5 cadenas (inglés y español), matriz de novedad con fecha,
  snowballing de 3-5 trabajos cercanos, tabla de 5-10 métodos afines. La novedad solo de aplicación no basta. Marca lo que no pudiste verificar.
- R2 Implementación propia: clase con BaseEstimator + ClassifierMixin (fit, predict, predict_proba), numpy/scipy. Prohibido usar xgboost, lightgbm, catboost,
  tabpfn, StackingClassifier o similares COMO NÚCLEO del modelo nuevo. Sin componentes neuronales. El núcleo (criterio de partición, regla de actualización,
  pérdida, representación o regla de combinación) debe explicarse línea a línea. Si hay miembros de librería en un ensamble, el aporte original debe
  medirse con ablación y debe ser significativo por sí mismo.
- R3 Superioridad frente a los 7 modelos base (XGBoost, Random Forest, SVM, KNN, Regresión Logística, Árbol de decisión, Naive Bayes), con ≥ 3 pruebas
  (p ajustado por Holm/BH < 0.05): una global tipo Friedman-Nemenyi, una pareada (t de Nadeau-Bengio o Wilcoxon) y otra (McNemar/DeLong); |Cliff δ| ≥ 0.147;
  y, si el docente lo exige, en conjuntos adicionales. En este proyecto se trabaja solo con el dataset de Taiwán (decisión del equipo).
  Las pruebas se fijan ANTES de ver resultados; elegir la que pasa después no es válido.
- Protocolo pre-registrado y congelado (hash/tag) antes de la corrida definitiva; semillas nuevas (ya usadas: 42, 1-4, 1000-1009, 2000-2004, 3000-3004,
  4000-4001, 5000-5004, 6000-6001; usar 7000+); desarrollo y confirmación separados; mismo presupuesto de búsqueda para todos; ≥ 10 semillas si el modelo es
  estocástico; el umbral se elige en el bucle interno para TODOS los modelos; reportar el número de diseños evaluados.
- Además: formulación matemática, pseudocódigo, una propiedad teórica, complejidad O(.), pruebas unitarias (aprende en sintéticos, semilla, entradas
  inválidas, clone), ablación, SHAP, calibración, análisis de errores.

## Qué debes entregar (en orden; para donde la evidencia lo indique)
1. Revisión de literatura 2024-2026 (con enlaces) de clasificación tabular no neuronal y de default crediticio. No afirmes novedad sin la matriz de búsqueda.
2. De 3 a 5 mecanismos candidatos **complementarios** a los GBDT, con: hipótesis de por qué darían puntajes distintos (correlación < 0.90 con XGB/CatBoost)
   sin perder AUC (≥ 0.78), formulación matemática, diferencia frente a los 5-10 trabajos más cercanos, riesgo para R1/R2, costo de implementación y efecto
   esperado en F1. Ideas a evaluar (no obligatorias): modelo aditivo con interacciones (tipo GAM/EBM boosting por variable, propio); representación por hojas
   de un bosque + modelo lineal regularizado; modelo por regímenes de PAY con compuerta aprendida (mezcla de expertos); codificación de secuencias de PAY
   (patrones de 6 meses como categorías con shrinkage jerárquico); partición con criterio nuevo robusto a ruido de etiquetas; boosting con pérdida
   que optimiza directamente un sustituto de F1; regla de combinación nueva (por ejemplo ponderar por régimen o por incertidumbre). Descarta lo que ya sea un
   parámetro de una librería o lo ya probado (sección 6).
3. **Piloto barato antes de implementar a fondo.** Para cada candidato mide en desarrollo (semillas 7000-7001): (a) AUC y F1 individuales, (b) correlación de
   Spearman con XGB y CatBoost, (c) ganancia de F1 de un ensamble simple (promedio en logit) con y sin el candidato frente a CatBoost solo y XGBoost@umbral.
   **Regla de parada fijada de antemano:** no se implementa el modelo completo si el candidato no cumple a la vez AUC ≥ 0.78, correlación < 0.90 y
   ganancia media ≥ +0.006 de F1 sobre XGBoost@umbral en desarrollo. Basta que una condición falle para parar.
4. Solo si el piloto pasa: implementación en numpy/scipy, pruebas unitarias y protocolo pre-registrado (hipótesis única principal, métrica, semillas nuevas,
   pruebas y corrección fijadas), congelado antes de la corrida confirmatoria.
5. Conclusión honesta: probabilidad estimada de lograr R3 y, si es baja, plan de cierre (empate documentado, análisis por régimen, tope de 4.0, y qué
   preguntarle al docente sobre qué prueba pareada se acepta).

## Reglas de integridad
- Explora en desarrollo, congela candidatos, confirma en semillas nuevas. No ajustes nada tras ver la confirmación. No busques "hasta que gane".
- Reporta todos los candidatos explorados y los negativos. No copies código de otros repositorios; todo debe defenderse en vivo.
- No uses información personal del equipo en consultas externas.
