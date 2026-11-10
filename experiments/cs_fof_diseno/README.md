# Pruebas de diseño de CS-FOF (modelo original)

Registro de las iteraciones de diseño, como pide el Entregable 3 (Figura 2: informar
cuántas iteraciones de diseño se evaluaron).

**Regla para no sobreajustar:** todas las decisiones de diseño se tomaron **solo con el
entrenamiento del pliegue externo 1** (validación interna estratificada de 3 pliegues,
semilla 42). Los pliegues externos de prueba no se usaron para decidir nada: se reservan
para la evaluación final 5 x 5 (`src/run_cs_fof_5x5.py`).

| Archivo | Qué es |
|---|---|
| `datos.py` | Réplica del preprocesamiento de `src/preprocessing.py` (mismas 25 columnas tras el one-hot y mismos pliegues externos). |
| `diseno.py` -> `ronda1_resultados.*` | Ronda 1: extensiones del corte oblicuo (más variables, 2 candidatos, más profundidad, más bins). |
| `diseno_r2.py` -> `ronda2_resultados.*` | Ronda 2: regularización (profundidad, hoja mínima, ε, max_features, número de árboles). |
| `anidada_random.py` -> `anidada_random.*` | Validación anidada 5 x 5 de referencia con búsqueda aleatoria (25 configuraciones) y `class_weight`. |

## Conclusiones de las 14 variantes (F1 de la clase default, validación interna)

- La versión base (un candidato oblicuo con las mismas q = √p variables) obtuvo **0.549**, frente a
  **0.545** de su ablación sin cortes oblicuos, **0.545** de Random Forest y **0.541** de XGBoost
  (ambos con sus mejores hiperparámetros del pliegue 1).
- Hacer el corte oblicuo más "fuerte" **empeoró** el modelo: con p/2 variables 0.545, con
  profundidad 14 y hoja 10 0.524. Interpretación: los árboles sobreajustan y el bosque
  pierde diversidad. Por eso se fijaron `n_oblique=1`, `oblique_features=None` y `max_depth <= 14`.
- Las variantes de regularización quedaron todas entre **0.546 y 0.550**: el modelo no
  depende de un ajuste fino de sus hiperparámetros.

Para reproducir: `python experiments/cs_fof_diseno/diseno.py` (requiere el `.xls` en `data/raw/`).

## Validación anidada 5 x 5 de referencia (búsqueda aleatoria, `class_weight`)

Mismos 5 pliegues externos que la tabla maestra, 5 internos, 25 configuraciones, métrica F1.
No reemplaza a `src/run_cs_fof_5x5.py` (que corre las 16 combinaciones), pero ya es comparable
pliegue a pliegue con los modelos base:

| Modelo | F1 (media ± d.e.) | AUC-ROC | Diferencia de F1 | Gana en | p Wilcoxon | δ de Cliff |
|---|---|---|---|---|---|---|
| **CS-FOF** (class_weight, aleatoria) | **0.5472 ± 0.0073** | **0.7846** |, |, |, |, |
| XGBoost (class_weight, Optuna), mejor global | 0.5466 ± 0.0091 | 0.7828 | +0.0006 | 4 de 5 | 0.625 | +0.20 |
| Random Forest (class_weight, aleatoria) | 0.5452 ± 0.0077 | 0.7831 | +0.0019 | 3 de 5 | 0.312 | +0.20 |
| XGBoost (class_weight, aleatoria) | 0.5414 ± 0.0107 | 0.7801 | +0.0058 | 5 de 5 | 0.062 | +0.44 |

CS-FOF queda primero en F1 medio y en AUC-ROC, pero con 5 pliegues las diferencias no son
estadísticamente significativas (con n = 5, Wilcoxon bilateral no puede bajar de p = 0.0625).
