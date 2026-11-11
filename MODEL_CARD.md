# Ficha del modelo: CS-FOF

**Cost-Sensitive Fisher Oblique Forest**, modelo original del Proyecto Final de Machine Learning
(Universidad del Norte). Autoras: Valeria Florez Sarmiento y Katherin Barrera López. Versión 3.0.0, noviembre de 2026.

## Descripción
Bosque aleatorio en el que cada nodo puede dividir a los clientes por una **combinación lineal de
variables**. La dirección de esa combinación es la dirección discriminante de Fisher, ponderada
por el costo del incumplimiento y calculada solo con los clientes que llegan al nodo. Compite con
los cortes por una sola variable y el nodo conserva el de menor impureza de Gini ponderada.
Implementación propia con numpy: `src/original_model.py`. Artefacto: `artifacts/cs_fof.joblib`
(hiperparámetros, versiones y esquema de entrada en `artifacts/metadata.json`).

## Uso previsto
- **Para qué:** estimar la probabilidad de que un cliente de tarjeta de crédito incumpla el pago del
  mes siguiente, como **apoyo** a la gestión de riesgo (priorizar cobranza preventiva, revisar cupos).
- **Quién:** analistas de riesgo y estudiantes, con fines académicos y demostrativos.
- **Salida:** puntaje de incumplimiento, clase predicha según un umbral y conjunto de predicción
  conforme al 90 %. La validación usa el umbral 0.5; en el tablero el umbral es ajustable y arranca en
  el de máximo F1 (52 %).

## Usos no recomendados
- Tomar decisiones automáticas sobre personas (negar crédito, subir tasas) **sin revisión humana**.
- Aplicarlo a otros países, épocas o productos sin reentrenarlo y validarlo: los datos son de un
  banco de Taiwán en 2005.
- Usar las variables `SEX`, `EDUCATION`, `MARRIAGE` o `AGE` para justificar decisiones: el modelo
  las usa como predictoras, lo que puede reproducir sesgos (ver Riesgos).

## Datos
*Default of Credit Card Clients* (UCI), 30 000 clientes, abril–septiembre de 2005; tras eliminar
35 duplicados, 29 965 clientes y 20 variables procesadas. 22.1 % de incumplimiento. Detalle en
`DATA_STATEMENT.md`.

## Métricas (validación anidada 5 × 5, mismos pliegues para todos los modelos)
| Modelo (mejor combinación) | F1 clase default | AUC-ROC |
|---|---|---|
| CatBoost (class_weight, búsqueda aleatoria) | 0.5480 ± 0.0088 | 0.7838 |
| **CS-FOF** (class_weight, búsqueda aleatoria) | **0.5479 ± 0.0054** | **0.7847** |
| XGBoost (class_weight, Optuna) | 0.5466 ± 0.0091 | 0.7828 |
| LightGBM (class_weight, Optuna) | 0.5462 ± 0.0086 | 0.7843 |
| Random Forest (class_weight, aleatoria) | 0.5452 ± 0.0077 | 0.7831 |

Desviaciones con `ddof=1`. CS-FOF es significativamente mejor que SVM, árbol de decisión, regresión
logística y Naive Bayes (t corregida de Nadeau–Bengio con Holm, p < 0.05), pero **no** que CatBoost,
XGBoost, LightGBM ni Random Forest. Probabilidades fuera de pliegue: Brier 0.168 y ECE 0.180, las
mejores de los cinco modelos de la tabla.

**Otros conjuntos de crédito** (F1, class_weight y búsqueda aleatoria): German 0.589 (CatBoost 0.605),
Australian 0.874 (XGBoost 0.880) y Polish año 1 0.430 (CatBoost 0.733).

## Limitaciones
- No es superior al mejor modelo de referencia y no es consistente en otros conjuntos: con desbalance
  extremo (Polish, 4 % de positivos) rinde mucho menos que los métodos de *boosting*.
- Solo se evaluó con la técnica de pesos de clase.
- Entrenar con pesos de costo **sobreestima** la probabilidad de incumplimiento (probabilidad media
  0.40 frente a una tasa real de 0.22). La regresión isotónica ajustada fuera de pliegue lo corrige
  (ECE de 0.180 a 0.003), pero entonces su ventaja de calibración frente a los demás desaparece.
- Implementación en numpy, sin código compilado: unas 5 veces más lenta que CatBoost al entrenar
  (6.6 s frente a 1.4 s con 24 mil clientes y 4 núcleos) y unas 600 veces al predecir (253 ms por cada
  1,000 clientes). Sirve para el tablero, no para puntuar millones de clientes.
- Con 90 % de confianza, la predicción conforme deja a la mitad de los clientes en el conjunto
  {pago, impago}: el modelo no puede decidir con esa confianza en uno de cada dos casos.
- Falla más que CatBoost justo en el grupo más grande de la cartera, los clientes al día en
  septiembre (77 % de los clientes; F1 0.261 frente a 0.298).

## Riesgos y consideraciones éticas
- **Sesgo:** las tasas de incumplimiento difieren por sexo y nivel educativo en los datos; el modelo
  puede trasladar esas diferencias a sus predicciones. El análisis de errores por subgrupo (capítulo
  de evaluación ampliada) muestra diferencias de F1 pequeñas entre edades y niveles educativos, y un
  F1 0.02 menor en mujeres que en hombres (0.540 frente a 0.559), igual que en CatBoost. Antes de un
  uso real habría que repetir esa revisión con datos actuales y con métricas de equidad.
- **Falsos negativos** (morosos no detectados) generan pérdidas; **falsos positivos** pueden llevar a
  restringir crédito a clientes solventes. El umbral debe fijarse según los costos de la entidad.

## Interpretabilidad
Los coeficientes de cada corte oblicuo se pueden leer como reglas de riesgo (por ejemplo, cuánto
pesa el atraso reciente frente al del mes anterior).

- **SHAP global:** SHAP por permutación en 300 clientes de prueba, con el mismo explicador que para CatBoost.
  Las dos importancias se parecen mucho (Spearman 0.95). Las variables principales son PAY_0, el cupo y PAY_2.
  CS-FOF reparte más peso entre los meses de atraso (capítulo del modelo original).
- **SHAP local:** el tablero muestra, para cada cliente, valores de Shapley exactos por grupo de variables.
