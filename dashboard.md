# Dashboard y despliegue

El tablero interactivo está desarrollado con Dash y Plotly y tiene las secciones **Contexto**, **EDA** y
**Modelos**, más un formulario de predicción en vivo.

- **URL pública (Render):** *pendiente de completar con la URL definitiva*.
- **Repositorio del tablero:** https://github.com/KathBarrera/dash_credit_kv
- **Despliegue:** archivo `render.yaml` (plan gratuito, 1 proceso de `gunicorn`). Los modelos se cargan
  una sola vez al iniciar desde archivos serializados; el tablero nunca reentrena.

La sección **Modelos** muestra la mejor combinación de cada clasificador, el diagnóstico del mejor modelo
y la subsección del modelo original (CS-FOF): su idea, la tabla de pruebas de hipótesis con p-valores
ajustados y el veredicto de superioridad del capítulo de validación.

El artefacto de CS-FOF (`artifacts/cs_fof.joblib`) incluye la codificación de variables y el modelo, y
su `artifacts/metadata.json` documenta las versiones de las librerías, la semilla, los hiperparámetros
finales, el esquema de columnas de entrada con sus rangos y la prueba de ida y vuelta.
