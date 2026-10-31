# Trazabilidad de correcciones

Este capítulo lista las observaciones del docente sobre el entregable anterior, qué se hizo con cada una, dónde está en el libro y qué evidencia la respalda. Las observaciones que ya estaban resueltas (la validación anidada 3 por 3, el F1 como métrica principal, la calibración isotónica y la comparación estadística) no tuvieron cambios y no se repiten aquí.

## Tabla de trazabilidad

| N.° | Observación del docente | Acción realizada | Ubicación | Evidencia |
|---|---|---|---|---|
| 1 | Faltan las curvas anytime. | Los cuatro optimizadores ahora guardan el historial de evaluaciones (score interno, costo y mejor score hasta ese momento). Con ese historial se repitió la búsqueda de tres modelos sobre los mismos folds externos y se graficó el arrepentimiento frente al número de evaluaciones y frente al costo. | `notebooks/09_anytime_multifidelidad.ipynb`, secciones 1 a 3 | Figuras `anytime_vs_evaluations.png` y `anytime_vs_cost.png`; código en `src/optimization.py` (`build_trace`) y `src/anytime_study.py`; datos en `results/anytime/` |
| 2 | Falta la multi-fidelidad. | Se implementó Hyperband desde cero (fidelidad igual a la fracción de filas de entrenamiento) y se comparó con Random Search sobre el mismo espacio, los mismos folds y un costo comparable. | `notebooks/09_anytime_multifidelidad.ipynb`, sección 4 | Código en `src/multifidelity.py`; tabla de la sección 4 con F1 externo y costo por corrida |
| 3 | No se entrega la tabla maestra. | `results/experiments_master.csv` estaba en `.gitignore` y no estaba en el repositorio. Se quitó del `.gitignore` y se agregó `results/master_table_112.csv`, que une las 112 combinaciones con F1 y métricas complementarias (media y desviación con `ddof=1`), semilla, tiempo, origen del cálculo y rango medio. Se agregó `results/best_per_model.csv` con la mejor combinación de cada modelo y el motivo de la selección. | `notebooks/08_tabla_maestra_svm.ipynb`, secciones 1 y 2 | `results/master_table_112.csv`, `results/best_per_model.csv`; código en `src/master_table.py` |
| 4 | El SVM de Colab no es comparable. | Se reajustaron las 16 filas con `sklearn.svm.SVC` en CPU, sobre los mismos folds y con los hiperparámetros que eligió Colab, y se midió la diferencia de F1 por fold. Además se repitió la comparación de Friedman con el SVM de Colab, con el SVM reajustado y sin SVM. | `notebooks/08_tabla_maestra_svm.ipynb`, secciones 3 y 4 | `results/svm_local_refit_check.csv`; código en `src/svm_comparability.py` |
| 5 | SHAP y LIME se apoyan en un solo fold y una sola observación. | SHAP global se repitió en los 3 folds externos (cada uno con los hiperparámetros de su propio bucle interno) para Random Forest y XGBoost, y LIME se comparó con SHAP en 120 observaciones por modelo, tomadas al azar de cada celda de la matriz de confusión, con 3 semillas de LIME. | `notebooks/06_interpretability.ipynb`, secciones 5 y 6 | Figuras `shap_importance_by_fold.png` y `lime_vs_shap_agreement.png`; datos en `results/interpretability/`; código en `src/interpretability_multi.py` |

## Resultados que cambiaron

Las correcciones recalcularon o agregaron resultados, y algunos son desfavorables para la versión anterior. Se reportan todos.

- **SVM quedaba subestimado.** Con los mismos hiperparámetros, los mismos folds y el mismo pipeline, `sklearn.svm.SVC` da un F1 mayor que el de cuML en las 16 filas (0.008 más en promedio, entre 0.004 y 0.017; 0.014 sin balanceo). La mejor combinación de SVM pasa de 0.5385 a 0.5442 y queda a menos de 0.001 de XGBoost. Random Forest sigue siendo la mejor combinación global y las conclusiones estadísticas (Friedman significativo, Nemenyi sin potencia) no cambian. Detalle en `08_tabla_maestra_svm.ipynb`. La tabla maestra conserva los valores de Colab como se reportaron y agrega el reajuste en columnas aparte.
- **La desviación estándar de la tabla original usaba `ddof=0`.** En `master_table_112.csv` todas las desviaciones usan `ddof=1`, que con 3 folds es la convención correcta. Las columnas `*_std` de `experiments_master.csv` no se modificaron.
- **Hyperband no demostró ventaja sobre Random Search.** F1 externo 0.0024 mayor, con un costo 1.28 veces mayor y sin llegar antes al objetivo; la diferencia es menor que la variación entre folds.
- **La explicación SHAP de XGBoost es menos estable que la de Random Forest.** El top 5 cambia entre folds (Jaccard medio 0.619), y solo `PAY_0`, `LIMIT_BAL_LOG` y `BILL_AMT_MEAN_6M_LOG` están siempre. LIME y SHAP coinciden casi siempre en el signo (más del 99 %) pero solo comparten entre 3 y 4 de sus 5 variables principales.
- **El ranking de las demás combinaciones no cambió.**

## Auditoría de fuga de información

Cada punto se verificó contra el código, no solo contra la documentación.

| Fuente posible de fuga | Qué se hace | Dónde se ve |
|---|---|---|
| Preprocesamiento ajustado con todo el conjunto | `OneHotEncoder` y `StandardScaler` están dentro del `Pipeline` (`ColumnTransformer` en `src/preprocessing.py`) y se ajustan solo con las filas de entrenamiento de cada fold. | `src/balancing.py::build_pipeline` |
| Balanceo aplicado antes de partir | SMOTE y ADASYN son pasos de `imblearn.pipeline.Pipeline`: solo remuestrean el conjunto que recibe `fit`, nunca el de validación ni el de prueba. `class_weight` calcula los pesos con el `y` de entrenamiento. | `src/balancing.py` |
| Selección de hiperparámetros con datos de prueba | Validación anidada: el bucle interno solo ve el fold de entrenamiento externo, y las métricas reportadas salen del fold externo de prueba. | `src/nested_cv.py` |
| Variables que codifican la respuesta | Las transformaciones del dataset final se calculan fila por fila y no aprenden parámetros: logaritmo con signo de `LIMIT_BAL` y `PAY_AMT*`, y media, pendiente y conteo de meses con saldo a favor de `BILL_AMT1` a `BILL_AMT6`. No hay codificación por la media del objetivo. | `src/preprocessing.py::add_bill_features`, `apply_log_transform` |
| Duplicados entre particiones | Las 35 copias exactas se eliminan una sola vez antes de partir (`remove_duplicate_rows`), de modo que una fila y su copia no quedan repartidas entre entrenamiento y prueba. | `src/preprocessing.py`, `notebooks/00_eda.ipynb` |
| Mezcla temporal o espacial | El conjunto es un corte transversal de un solo periodo y no tiene identificador de cliente repetido (el `ID` se eliminó), así que no hay mezcla temporal que controlar. Se declara como limitación: no se puede verificar que un mismo cliente no aparezca en dos filas bajo valores distintos. | `src/config.py`, `notebooks/00_eda.ipynb` |
| Interpretabilidad y estudios anytime | SHAP y LIME usan el pipeline reajustado con los hiperparámetros del fold y explican filas de ese mismo fold de prueba. El estudio anytime optimiza solo con el fold de entrenamiento externo y usa el de prueba una vez, al final. | `src/interpretability_multi.py`, `src/anytime_study.py` |

Un punto que no es fuga pero conviene decir: los F1 y las métricas complementarias de los seis modelos locales se reprodujeron con los hiperparámetros guardados y coinciden con la tabla maestra (columna `f1_check_max_abs_diff` de `results/experiments_posthoc_metrics.csv`, igual a 0 en las 96 filas).
