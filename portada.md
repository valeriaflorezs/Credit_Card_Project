# Predicción del Incumplimiento de Pago en Tarjetas de Crédito usando Machine Learning

**Autoras:** Katherin Barrera López y Valeria Florez Sarmiento

Departamento de Matemáticas, Física y Ciencia de Datos, Universidad del Norte, Barranquilla, Colombia

Contacto: lkatherin@uninorte.edu.co y florezvaleria@uninorte.edu.co

Tablero interactivo del EDA: <https://dash-credit-kv.onrender.com>

---

## Resumen

El incumplimiento de pago en tarjetas de crédito genera pérdidas, costos de cobranza y exigencias de capital para las
entidades financieras; anticiparlo permite ajustar cupos, priorizar la cobranza y reducir la exposición al riesgo. En
el conjunto de datos de clientes de Taiwán se han usado modelos estadísticos, ensambles de árboles, remuestreo y redes
profundas; sus resultados son muy similares, las redes pierden interpretabilidad y los árboles solo dividen el espacio
en paralelo a los ejes. Se propone el *Cost-Sensitive Fisher Oblique Forest* (CS-FOF), un bosque aleatorio
implementado desde cero cuyos nodos pueden cortar por la dirección discriminante de Fisher ponderada por el costo del
incumplimiento; se basa en los árboles oblicuos y se evalúa con validación cruzada anidada y pruebas de hipótesis. En
un experimento piloto, CS-FOF obtuvo el mayor F1 de la clase *default* entre seis modelos (0.547 frente a 0.546, una
mejora relativa de 0.3%) con un AUC-ROC de 0.780; se espera que la validación final confirme esta ventaja de forma
significativa. El aporte es un clasificador interpretable y no neuronal que lleva los cortes oblicuos al riesgo
crediticio, con garantías teóricas sobre la calidad de cada partición.

**Palabras clave:** riesgo crediticio, incumplimiento de pago, tarjetas de crédito, árboles de decisión oblicuos,
bosques aleatorios, discriminante de Fisher, aprendizaje sensible al costo, desbalance de clases.

### Resultados en cifras

| Modelo | F1 de la clase *default* |
|---|---|
| Regresión logística (línea base, partición 80/20) | cerca de 0.38 |
| Mejores combinaciones con balanceo (validación anidada 5 x 5) | entre 0.52 y 0.55 |
| XGBoost con `class_weight` y optimización bayesiana | 0.547 |

- Esa mejora proviene en buena parte del balanceo, que desplaza el umbral de decisión; el ROC-AUC casi no cambia
  entre técnicas.
- Con 5 folds, la prueba post-hoc de Nemenyi solo separa combinaciones muy alejadas en el ranking (224 de 6,216
  pares); entre las combinaciones de cabeza no hay diferencias significativas, por lo que esa comparación es
  descriptiva.
- La interpretabilidad se aborda con SHAP en los 5 folds (Random Forest y XGBoost) y una comparación LIME frente a
  SHAP.

### Contenido del libro

1. **Marco del proyecto:** introducción, planteamiento del problema, hipótesis, justificación, objetivos, marco
   teórico y metodología.
2. **Análisis exploratorio (EDA)** y construcción del dataset final.
3. **Modelo base:** regresión logística con partición 80/20.
4. **Pipeline y experimentos:** desarrollo del pipeline y corrida de las combinaciones.
5. **Análisis de resultados:** tabla de las 112 combinaciones, comparación estadística, calibración y complejidad.
6. **Interpretabilidad:** SHAP y LIME.
7. **Modelo original:** CS-FOF, LightGBM y CatBoost frente al mejor F1 de las 112 combinaciones.
