# Evaluación de preprocesamiento, análisis exploratorio y comparación sistemática de modelos para la predicción de incumplimiento de pago en tarjetas de crédito

**Autoras:** Katherin Barrera López y Valeria Florez Sarmiento

Departamento de Matemáticas, Física y Ciencia de Datos, Universidad del Norte, Barranquilla, Colombia

Contacto: lkatherin@uninorte.edu.co y florezvaleria@uninorte.edu.co

Tablero interactivo del EDA: <https://dash-credit-kv.onrender.com>

---

## Resumen del proyecto

Este trabajo presenta un flujo completo de aprendizaje automático para la predicción del incumplimiento de pago
(*default*) de clientes de tarjetas de crédito, a partir del conjunto de datos público *Default of Credit Card
Clients* del repositorio UCI, compuesto originalmente por 30,000 observaciones y 23 variables predictoras. El estudio
integra el proceso de extracción, transformación y carga (ETL), un análisis exploratorio de datos (EDA) y una
evaluación sistemática de modelos de clasificación.

El EDA incluyó estadística descriptiva, diagnóstico de valores faltantes y duplicados, análisis de la forma de las
distribuciones, detección de valores atípicos, comparación formal de grupos con pruebas no paramétricas y de
independencia, y evaluación de multicolinealidad con correlaciones y el Factor de Inflación de la Varianza (VIF).
Se eliminaron 35 registros duplicados exactos (quedaron 29,965), se encontraron categorías no oficiales en
`EDUCATION` y `MARRIAGE`, y se halló una multicolinealidad severa entre las variables de facturación mensual, que se
resumieron en tres variables derivadas. La variable `PAY_0` destacó como la señal individual más discriminante. La
variable objetivo presenta un desbalance moderado: 77.9 % sin *default* y 22.1 % con *default*.

Como línea base se entrenó una regresión logística con partición estratificada 80/20, cuyo F1 para la clase
*default* es de cerca de 0.38. Después se evaluaron 112 configuraciones mediante un diseño factorial (7 modelos por 4
técnicas de balanceo por 4 métodos de optimización de hiperparámetros), con validación cruzada anidada (5 folds
externos por 5 internos) y el F1 de la clase *default* como métrica principal. La mejor configuración fue XGBoost con
ponderación de clases y optimización bayesiana (F1 medio de 0.547), seguida de cerca por Random Forest con
ponderación de clases.
Con ese F1 como referencia, el proyecto propone además un modelo original (un bosque aleatorio con cortes oblicuos
sensibles al costo) y lo contrasta con dos métodos recientes de boosting.

**Palabras clave:** aprendizaje automático, análisis exploratorio de datos, preprocesamiento, clasificación binaria,
riesgo crediticio, desbalance de clases, validación cruzada anidada, optimización de hiperparámetros, F1-score.

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
