#  Default of Credit Card Clients

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Conda](https://img.shields.io/badge/Environment-Conda-44A833?style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

> **Resumen del Proyecto:** Este proyecto desarrolla un flujo completo e integral de ciencia de datos y aprendizaje automático para predecir el riesgo de incumplimiento de pago (*default*) en clientes de tarjetas de crédito. A partir del análisis del conjunto de datos del repositorio UCI (30,000 registros), la solución abarca desde un pipeline riguroso de extracción, limpieza y transformación de datos (ETL) y un profundo análisis exploratorio (EDA) orientado a solucionar problemas críticos de desbalance de clases y multicolinealidad severa entre montos facturados, hasta un diseño experimental factorial exhaustivo de 112 configuraciones. La estrategia evalúa sistemáticamente algoritmos de clasificación, representaciones de variables, técnicas de balanceo y optimizaciones avanzadas (algoritmos genéticos y bayesianos), logrando elevar significativamente el *F1* de la clase minoritaria y reducir los falsos negativos frente al modelo de regresión logística base.

---

## Inicio Rápido (Requisitos y Configuración)

Asegúrate de contar con **Anaconda** o **Miniconda** instalado en tu equipo antes de comenzar.

<a href="https://www.anaconda.com/download/success?reg=skipped-miniconda" target="_blank">
  <img src="https://img.shields.io/badge/Descargar_Miniconda-44A833?style=for-the-badge&logo=anaconda&logoColor=white" alt="Descargar Miniconda" />
</a>

<details open>
<summary><b> Pasos para la Configuración del Entorno Virtual (Anaconda Prompt)</b></summary>

<br>

1. **Abrir Anaconda Prompt** desde el menú de inicio de tu sistema operativo.
2. **Crear el entorno virtual** con Python 3.12:
   ```bash
   conda create --name credit_card_env python=3.12 -y

```

3. **Activar el entorno virtual**:
```bash
conda activate credit_card_env

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
├── 📁 data/                  # Datos del proyecto
│   └── processed/            # Conjuntos de datos procesados y transformados
│
├── 📁 logs/                  # Registros de ejecución y monitoreo
├── 📁 notebooks/             # Jupyter Notebooks de análisis exploratorio y modelado
├── 📁 results/               # Resultados de experimentos, métricas y gráficas
├── 📁 src/                   # Código fuente modular reutilizable
│
├── .gitignore                # Archivos ignorados por Git
├── EXPERIMENT_STATUS.md      # Estado de experimentos y seguimiento de pruebas
├── README.md                 # Documentación principal del proyecto
├── requirements.txt          # Lista de dependencias del proyecto
├── _config.yml               # Configuración de Jupyter Book
└── _toc.yml                  # Tabla de contenidos de Jupyter Book

```

---

##  Flujo de Trabajo del Proyecto

---

##  Guía de Ejecución

1. **Activar el entorno virtual:**
```bash
conda activate credit_card_env

```


2. **Ejecución desde VS Code (Recomendado):**
* Abre la carpeta del proyecto en **VS Code**.
* Selecciona el intérprete de Python correspondiente al entorno `credit_card_env` (presiona `Ctrl + Shift + P` -> **Python: Select Interpreter**).
* Abre cualquier cuaderno dentro de la carpeta `notebooks/` y ejecútalo directamente.



---

## Autoras

* **Valeria Florez Sarmiento**
* **Katherin Barrera López**
