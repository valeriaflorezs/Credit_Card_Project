# Default of Credit Card Clients

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Conda](https://img.shields.io/badge/Environment-Conda-44A833?style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

> **Resumen del Proyecto:** Este proyecto desarrolla un flujo completo e integral de ciencia de datos y aprendizaje automático para predecir el riesgo de incumplimiento de pago (*default*) en clientes de tarjetas de crédito. A partir del conjunto de datos del repositorio UCI (30,000 registros; 29,965 y 21 columnas tras eliminar 35 duplicados exactos y transformar variables), la solución abarca un pipeline de extracción, limpieza y transformación de datos (ETL) y un análisis exploratorio (EDA) orientado al desbalance de clases (cerca del 22 % de *default*) y a la multicolinealidad entre montos facturados, hasta un diseño experimental factorial de 112 configuraciones: 7 modelos x 4 técnicas de balanceo x 4 métodos de optimización, con validación cruzada anidada (3 folds externos x 3 internos). Seis modelos se ejecutan localmente (96 combinaciones) y SVM en Colab con GPU (16 combinaciones, ver la nota de comparabilidad en `05_results_analysis.ipynb`).
>
> **Resultados y alcance:** el F1 de la clase minoritaria sube de cerca de 0.38 (regresión logística sin balanceo) a entre 0.52 y 0.55 con balanceo. Esa mejora proviene en buena parte del balanceo, que desplaza el umbral de decisión (el ROC-AUC casi no cambia entre técnicas); la comparación estadística entre las 112 combinaciones es descriptiva, porque con 3 folds el post-hoc de Nemenyi no puede declarar diferencias significativas.

---

## Inicio Rápido (Requisitos y Configuración)

Asegúrate de contar con **Anaconda** o **Miniconda** instalado en tu equipo antes de comenzar.

<a href="https://www.anaconda.com/download/success?reg=skipped-miniconda" target="_blank">
  <img src="https://img.shields.io/badge/Descargar_Miniconda-44A833?style=for-the-badge&logo=anaconda&logoColor=white" alt="Descargar Miniconda" />
</a>

<details open>
<summary><b>Pasos para la Configuración del Entorno Virtual (Anaconda Prompt)</b></summary>

<br>

1. **Abrir Anaconda Prompt** desde el menú de inicio de tu sistema operativo.
2. **Crear el entorno virtual** con Python 3.12:
   ```bash
   conda create --name creditcard python=3.12 -y

```

3. **Activar el entorno virtual**:
```bash
conda activate creditcard

```


4. **Instalar las dependencias**:
Navega a la carpeta raíz del repositorio e instala las librerías necesarias:
```bash
pip install -r requirements.txt

```



---

## Estructura del Repositorio

```text
Credit_Card_Project/
│
├── data/                     # Datos del proyecto
│   └── processed/            # Conjuntos de datos procesados y transformados
│
├── logs/                     # Registros de ejecución y monitoreo
├── notebooks/                # Jupyter Notebooks de análisis exploratorio y modelado
├── results/                  # Resultados de experimentos, métricas y gráficas
├── src/                      # Código fuente modular reutilizable
│
├── .gitignore                # Archivos ignorados por Git
├── EXPERIMENT_STATUS.md      # Estado de experimentos y seguimiento de pruebas
├── README.md                 # Documentación principal del proyecto
├── requirements.txt          # Lista de dependencias del proyecto
├── _config.yml               # Configuración de Jupyter Book
└── _toc.yml                  # Tabla de contenidos de Jupyter Book

```

---

## Flujo de Trabajo del Proyecto

| Notebook | Contenido |
|---|---|
| `00_eda.ipynb` | EDA y construcción de `data/processed/dataset_final.csv` (Parte 5) |
| `01_baseline_logistic.ipynb` | Baseline: regresión logística con partición 80/20 |
| `02_setup_check.ipynb`, `03_pipeline_dev.ipynb` | Verificación del entorno y desarrollo del pipeline (datos sintéticos) |
| `04_run_experiments.ipynb` | Corrida de las combinaciones (6 modelos, local) a `results/experiments_master.csv` |
| `07_svm_gpu_colab.ipynb` | SVM (16 combinaciones) en Colab con GPU a `results/svm_gpu_results_colab.csv` |
| `05_results_analysis.ipynb` | Tabla de las 112 combinaciones, métricas complementarias (ROC-AUC, PR-AUC; el F1 sigue siendo la principal), Friedman/Nemenyi, comparación por factor, DeLong, calibración y complejidad |
| `06_interpretability.ipynb` | SHAP del Random Forest y comparación LIME vs. SHAP de XGBoost (una observación) |

---

## Guía de Ejecución

1. **Activar el entorno virtual:**
```bash
conda activate creditcard

```


2. **Ejecución desde VS Code (Recomendado):**
* Abre la carpeta del proyecto en **VS Code**.
* Selecciona el intérprete de Python correspondiente al entorno `creditcard` (presiona `Ctrl + Shift + P` -> **Python: Select Interpreter**).
* Abre cualquier cuaderno dentro de la carpeta `notebooks/` y ejecútalo directamente.



---

## Autoras

* **Valeria Florez Sarmiento**
* **Katherin Barrera López**
