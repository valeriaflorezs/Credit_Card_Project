# Dashboard y despliegue

El tablero interactivo está desarrollado con Dash y Plotly. Tiene una página por tema, agrupadas en las tres
secciones que pide el Entregable 3: **Contexto** (Inicio, Introducción, Problema y Objetivos, Marco
Teórico), **EDA** (Univariado y Multivariado) y **Modelos** (Modelos, Modelo original, Predecir y
Conclusiones). Además incluye Enlaces, Equipo y un asistente conversacional (Kyo).

- **URL pública (Render):** https://dash-credit-kv.onrender.com
- **Repositorio del tablero:** https://github.com/KathBarrera/dash_credit_kv
- **Despliegue:**
  - `render.yaml`: plan gratuito, 1 proceso de `gunicorn` con 4 hilos.
  - `requirements.txt` con versiones exactas y Python 3.13.
  - Los modelos se cargan una sola vez al iniciar, desde archivos serializados con `joblib`; el tablero
    nunca reentrena ni busca hiperparámetros.
  - Con los tres modelos cargados usa unos 340 MB de los 512 MB del plan.

## Modelos

- **Tabla de líderes** (`dash_table.DataTable`) con las 148 combinaciones. Se puede ordenar y filtrar, y
  muestra modelo, balanceo, búsqueda, F1 como media ± desviación estándar del bucle externo, AUC-ROC,
  *recall*, precisión y tiempo de inferencia de los cinco mejores. Una estrella marca la mejor combinación
  de cada modelo.
- **Diagnóstico del mejor modelo:**
  - matriz de confusión fuera de pliegue;
  - mapa de calor de F1 por modelo y técnica de balanceo;
  - curvas anytime;
  - SHAP global en los 5 pliegues;
  - comparación entre LIME y SHAP.
- **Modelo original (CS-FOF):** página propia con cuatro pestañas.
  - **Teoría:** ejemplo visual del corte oblicuo frente a los cortes por ejes.
  - **Metodología:** formulación, pseudocódigo y propiedades.
  - **Código:** la clase real, con colores de sintaxis.
  - **Resultados:**
    - comparación animada con Random Forest;
    - tabla de pruebas de hipótesis con p-valores ajustados y tamaños del efecto;
    - conjuntos adicionales, robustez y calibración;
    - veredicto de superioridad.

## Predicción en vivo

El formulario se genera a partir de las 23 variables originales y valida rangos y tipos. Un **selector de
modelo** permite elegir entre tres modelos:

| Modelo | F1 en prueba | AUC-ROC en prueba | Cobertura conforme (nominal 90 %) |
|---|---|---|---|
| Random Forest | 0.533 | 0.774 | 89.4 % |
| CatBoost (mejor global) | 0.536 | 0.775 | 89.6 % |
| CS-FOF (modelo original) | 0.540 | 0.775 | 89.5 % |

Detalles de esta comparación:

- Los tres modelos usan la misma partición 80/20 (semilla 42), las mismas variables y pesos de clase.
- El Random Forest eligió sus hiperparámetros con búsqueda aleatoria sobre el 80 % de entrenamiento.
- CatBoost y CS-FOF usan los hiperparámetros finales de la validación anidada, sin búsqueda nueva.
- Las cifras son de una sola partición: sirven para mostrar el modelo en uso, no para compararlos. La
  comparación válida es la del capítulo de validación.

Para cada cliente, el tablero muestra:

- **Puntaje y veredicto:** la probabilidad estimada de incumplimiento y la decisión con umbral 0.5.
- **Conjunto de predicción conforme al 90 %:**
  - Se calcula por clase (Mondrian). Se calibró con la mitad de la partición de prueba y su cobertura se
    verificó con la otra mitad (tabla de arriba).
  - Si el conjunto contiene las dos clases, el modelo no puede decidir con esa confianza.
- **Explicación local SHAP:**
  - Son valores de Shapley exactos por grupo de variables: historial de pagos, montos pagados, facturación,
    cupo y perfil demográfico.
  - Con 5 grupos se evalúan las 32 coaliciones, cada una promediada sobre 40 clientes reales de fondo
    (SHAP intervencional).
  - El puntaje medio más los aportes da exactamente el puntaje del cliente.
  - Se agrupa porque las variables de un mismo grupo están muy correlacionadas.
- **Votos de los árboles** (en los bosques) y una curva **«¿Qué pasaría si…?»**, que muestra cómo cambia el
  puntaje al mover una sola variable.

## Artefactos

La carpeta `artifacts/` del proyecto contiene el pipeline final (preprocesamiento + balanceo + modelo) de
la mejor combinación de cada uno de los 10 modelos.

- **Archivos de cada modelo:** un `.joblib` comprimido y un `metadata_<modelo>.json`, que documenta:
  - versiones de las librerías y semilla;
  - hiperparámetros finales y cómo se eligieron;
  - esquema de columnas de entrada con sus rangos;
  - métricas del bucle externo;
  - resultado de la prueba de ida y vuelta.
  - En CS-FOF los archivos son `cs_fof.joblib` y `metadata.json`.
- **Cómo se fijaron los hiperparámetros:** los candidatos son las configuraciones que eligió el bucle
  interno en los 5 pliegues externos. Gana la de mayor F1 medio en una validación cruzada de 5 pliegues
  sobre todos los datos, y con ella se reentrena el pipeline con todos los datos (`src/train_artifacts.py`).
  CS-FOF repitió su búsqueda aleatoria sobre todos los datos (`src/train_cs_fof_artifact.py`).
- **Prueba de ida y vuelta:** `artifacts/ida_y_vuelta.npz` guarda 200 clientes fijos y las probabilidades
  del modelo en memoria. `tests/test_artifacts.py` comprueba en cada *push* que los modelos cargados
  predicen exactamente lo mismo.
