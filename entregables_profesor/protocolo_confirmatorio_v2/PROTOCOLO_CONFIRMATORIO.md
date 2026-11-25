# Protocolo confirmatorio pre-registrado (congelado antes de correr)

Modelo original propuesto: **HIB** (Hybrid Ensemble with soft-F1 weights). Regla de combinación propia: pesos w = softmax(theta) sobre los
logits de M modelos base, optimizados (L-BFGS-B, 60 iteraciones) para maximizar un F1 suave 2*sum(q*y)/(sum(q)+sum(y)), con
q = sigmoide((s - t)/tau), s = Z w, tau = 0.3, penalización 0.05*M*||w - 1/M||^2, sobre predicciones OOF internas. Umbral final: máximo F1
duro sobre el puntaje OOF (cortes entre 5 % y 60 % de positivos predichos). Sin sensibilidad al costo en la regla.
Miembros (8): cs_fof, xgb, cat, lgbm, rf, et, hgb, lr. Variante de ablación HIB_sin: los mismos sin cs_fof.

## Problema principal (Taiwán)
- Datos: dataset_final.csv (29 965 filas) + variables derivadas del .xls (utilización, razón pago/factura, meses con pago mínimo, racha de atraso, etc.),
  iguales para TODOS los modelos. NaN estructurales de razones = -1.
- Particiones: 5 repeticiones x 5 pliegues estratificados, semillas 5000-5004 (25 pliegues externos). Predicciones OOF internas: 3 pliegues.
- Benchmarks (los 7 del Entregable 2): XGBoost, Random Forest, SVM (RBF), KNN, Regresión Logística, Árbol de decisión, Naive Bayes, con los
  hiperparámetros de artifacts/ y class_weight balanced (SVM: balanced; NB: priors por defecto), mismo umbral interno para todos (SVM: decision_function).
- Métrica principal: F1 de la clase default con umbral elegido en el bucle interno. Secundarias: AUC, PR-AUC.
- Hipótesis principal H1 (7 contrastes): HIB tiene mayor F1 que cada benchmark. Ablación H2 (exploratoria): HIB vs HIB_sin.
- Pruebas (todas sobre las diferencias pareadas por pliegue, Holm sobre los 7 contrastes de cada prueba):
  T1 t pareada con corrección de Nadeau-Bengio (1/J + n_test/n_train); T2 Wilcoxon de rangos con signo; T3 Friedman global sobre los 8 modelos
  con post-hoc de Nemenyi. Tamaño de efecto: delta de Cliff.
- Criterio de superioridad (guía 6.4): media de HIB mayor que la de cada benchmark; las tres pruebas con p ajustado < 0.05 frente a cada benchmark;
  |delta de Cliff| >= 0.147. Se declara de antemano que T1 probablemente no rechazará frente a XGBoost y RF.

## Reglas
- Una sola corrida (solo Taiwán). Nada se ajusta tras ver resultados. Se reportan todos los resultados, incluidos los desfavorables.
- Iteraciones de diseño previas a este protocolo: 18 candidatos del cribado + 2 sondas + 24 variantes de Polish + 2 582 combinaciones (13 confirmadas)
  + 3 reglas/conjuntos en desarrollo (4000-4001) + pruebas de codificación categórica y de razones (6000-6001).
Congelado: 2026-11-25 04:25:21 UTC
