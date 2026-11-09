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
- **Salida:** probabilidad de incumplimiento y clase predicha con umbral 0.5.

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
  0.40 frente a una tasa real de 0.22). Si se usan como probabilidades reales, recalibrar (Platt o
  isotónica).
- Implementación en Python puro: entrenar es más lento que con librerías en C (unos 13 s por ajuste
  con 4 procesos, frente a 1–3 s de Random Forest y los métodos de *boosting*).

## Riesgos y consideraciones éticas
- **Sesgo:** las tasas de incumplimiento difieren por sexo y nivel educativo en los datos; el modelo
  puede trasladar esas diferencias a sus predicciones. Antes de cualquier uso real hay que medir
  el desempeño por subgrupo.
- **Falsos negativos** (morosos no detectados) generan pérdidas; **falsos positivos** pueden llevar a
  restringir crédito a clientes solventes. El umbral debe fijarse según los costos de la entidad.

## Interpretabilidad
Los coeficientes de cada corte oblicuo se pueden leer como reglas de riesgo (por ejemplo, cuánto
pesa el atraso reciente frente al del mes anterior). Para explicaciones globales y locales se usa
SHAP por permutación.
