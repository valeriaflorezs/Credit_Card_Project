# Reproducibilidad, archivo del repositorio y declaración de autoría

## Lista de verificación de reproducibilidad

```{list-table}
:header-rows: 1
:name: tab-reproducibilidad

* - Elemento
  - Detalle
* - Código
  - https://github.com/valeriaflorezs/Credit_Card_Project (licencia MIT)
* - Archivo con identificador persistente
  - DOI de Zenodo: *pendiente de completar con el DOI de la release v3.0.0*
* - Datos
  - UCI *Default of Credit Card Clients* (CC BY 4.0); ver `DATA_STATEMENT.md`
* - Semilla global
  - 42 (`src/config.py`), usada en particiones, búsquedas y modelos
* - Validación
  - Anidada estratificada, 5 pliegues externos × 5 internos
* - Ejecuciones
  - Una corrida por combinación; dispersión reportada como desviación estándar entre pliegues (`ddof=1`)
* - Rangos de hiperparámetros
  - `src/models.py` (`MODEL_REGISTRY`), uno por modelo
* - Presupuesto de búsqueda
  - Malla ≤ 200 combinaciones, 25 aleatorias, 30 de Optuna, genético 16 × 10 (`src/config.py`)
* - Software
  - Python 3.12, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10, optuna 5.0.0, deap 1.4.3 (`requirements.txt`)
* - Hardware
  - Windows, AMD Ryzen 7, 16 GB de RAM, sin GPU (modelos de árboles, CS-FOF y boosting); Google Colab con GPU T4 (SVM y KNN)
* - Tiempo de cómputo de CS-FOF
  - Malla 2.7 h, aleatoria 2.5 h y Optuna 2.5 h (5 × 5, 8 procesos)
* - Pruebas automáticas
  - `python -m pytest tests/ -q` (14 pruebas) en GitHub Actions en cada *push*
* - Modelo final
  - `python -m src.train_cs_fof_artifact` → `artifacts/cs_fof.joblib` y `artifacts/metadata.json`
* - Corridas
  - `python -u -m src.run_cs_fof_5x5 class_weight` y `python -u -m src.run_boosting_recientes_5x5 --tecnicas class_weight --metodos grid_search random_search bayesian_optuna`
```

## Declaración de autoría y uso de herramientas

Este trabajo fue realizado por **Valeria Florez Sarmiento** y **Katherin Barrera López**, quienes
contribuyeron de manera conjunta al diseño, la ejecución de los experimentos, el análisis y la
redacción.

Durante el proyecto se utilizó la herramienta de inteligencia artificial generativa **Claude
(Anthropic)** como apoyo en la búsqueda bibliográfica, la programación (incluida una implementación de
referencia del modelo original y sus pruebas), el análisis estadístico y la redacción. Todo el contenido
generado con apoyo de la herramienta fue revisado, verificado y editado por las autoras, quienes
comprenden el código entregado, pueden explicarlo y modificarlo, y asumen la total responsabilidad por
el contenido, los análisis y las conclusiones de este informe. La herramienta no figura como autora.

Firmado:

| Nombre | Firma | Fecha |
|---|---|---|
| Valeria Florez Sarmiento | | |
| Katherin Barrera López | | |
