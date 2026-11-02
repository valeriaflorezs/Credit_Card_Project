# Default of Credit Card Clients

## Evaluación de preprocesamiento, análisis exploratorio y comparación sistemática de modelos para la predicción de incumplimiento de pago en tarjetas de crédito

**Autoras:** Katherin Barrera López y Valeria Florez Sarmiento

Departamento de Matemáticas, Física y Ciencia de Datos, Universidad del Norte, Barranquilla, Colombia

Contacto: lkatherin@uninorte.edu.co y florezvaleria@uninorte.edu.co

Tablero interactivo del EDA: <https://dash-credit-kv.onrender.com>

---

## Resumen del proyecto

Este proyecto desarrolla un flujo completo de ciencia de datos y aprendizaje automático para predecir el
incumplimiento de pago (*default*) de clientes de tarjetas de crédito. Parte del conjunto de datos público
*Default of Credit Card Clients* del repositorio UCI (30,000 clientes de Taiwán, 2005; 29,965 registros y 21
columnas tras eliminar 35 duplicados exactos y transformar variables) y recorre el proceso de extracción,
limpieza y transformación (ETL), el análisis exploratorio (EDA) y la comparación sistemática de modelos.

El EDA identificó un desbalance moderado de clases (cerca del 22 % de *default*), categorías no documentadas en
`EDUCATION` y `MARRIAGE`, y una multicolinealidad severa entre los montos facturados, que se resumieron en
tres variables derivadas. El estado de pago más reciente (`PAY_0`) resultó la señal individual más
discriminante.

La comparación de modelos se hizo con un diseño factorial de **112 configuraciones**: 7 modelos x 4 técnicas de
balanceo x 4 métodos de optimización de hiperparámetros, evaluadas con validación cruzada anidada (5 folds
externos x 5 internos) y con el F1 de la clase minoritaria como métrica principal.

## Resultados principales

- El F1 de la clase *default* sube de cerca de 0.38 (regresión logística sin balanceo) a entre 0.52 y 0.55 con balanceo.
- Esa mejora proviene en buena parte del balanceo, que desplaza el umbral de decisión; el ROC-AUC casi no
  cambia entre técnicas.
- Con 5 folds, la prueba post-hoc de Nemenyi solo separa combinaciones muy alejadas en el ranking (224 de
  6,216 pares); entre las combinaciones de cabeza no hay diferencias significativas, por lo que esa comparación
  es descriptiva.
- La interpretabilidad se aborda con SHAP en los 5 folds (Random Forest y XGBoost) y una comparación LIME vs. SHAP.

## Contenido de este libro

1. **Introducción:** contexto, planteamiento del problema, justificación, objetivos y metodología.
2. **Análisis exploratorio (EDA)** y construcción del dataset final.
3. **Modelo base:** regresión logística con partición 80/20.
4. **Pipeline y experimentos:** desarrollo del pipeline y corrida de las combinaciones.
5. **Análisis de resultados:** tabla de las 112 combinaciones, comparación estadística, calibración y complejidad.
6. **Interpretabilidad:** SHAP y LIME.

**Palabras clave:** aprendizaje automático, análisis exploratorio de datos, riesgo crediticio, clasificación
binaria, desbalance de clases, optimización de hiperparámetros, validación cruzada anidada, F1-score.
