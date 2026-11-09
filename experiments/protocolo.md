# Protocolo experimental del modelo original (CS-FOF)

> **Declaración de desviación.** El Entregable 3 pide congelar este protocolo con un tag de Git
> **antes** de los experimentos definitivos. Este documento se escribió el **8 de noviembre de 2026,
> después** de ejecutarlos. No se modificó ninguna fecha. Las decisiones que describe sí se tomaron
> antes de ver los pliegues de prueba: el diseño se fijó solo con el entrenamiento del pliegue
> externo 1 (`experiments/cs_fof_diseno/`), y el protocolo de evaluación es el mismo del
> Entregable 2. Los análisis que no estaban previstos se marcan como **exploratorios**.

## 1. Hipótesis
- **H1 (principal).** CS-FOF obtiene un F1 medio de la clase *default* mayor que la mejor combinación
  de cada uno de los 7 modelos base y de los 2 métodos recientes (LightGBM y CatBoost).
- **H2.** El corte oblicuo mejora el F1 frente al mismo bosque sin cortes oblicuos (ablación).
- **H3 (exploratoria).** La ponderación por costo mejora el F1 frente a la versión sin costo.

## 2. Métricas
- **Principal:** F1 de la clase *default* (clase positiva = incumple), umbral 0.5.
- **Secundarias:** precisión, *recall*, AUC-ROC, AUC-PR y *accuracy*.

## 3. Datos y particiones
- *Default of Credit Card Clients* (UCI), 29 965 clientes tras eliminar duplicados, 20 variables
  (ver `DATA_STATEMENT.md`).
- Validación cruzada anidada estratificada: 5 pliegues externos × 5 internos, `shuffle=True`,
  semilla global 42 (`src/config.py`). Mismas particiones para todos los modelos.
- **Conjuntos adicionales previstos:** German Credit, Australian Credit Approval y Give Me Some
  Credit. **No se ejecutaron** por tiempo de cómputo (ver sección 7).

## 4. Modelos y presupuesto
| Modelo | Técnicas de balanceo | Métodos de búsqueda |
|---|---|---|
| 7 modelos base (Entregable 2) | none, class_weight, smote, adasyn | malla, aleatoria, Optuna, genético |
| LightGBM, CatBoost | class_weight | malla, aleatoria, Optuna |
| CS-FOF | class_weight | malla, aleatoria, Optuna, genético |

Mismo presupuesto por método para todos: malla de hasta 200 combinaciones (32 para Random Forest y
CS-FOF; 64 para XGBoost, LightGBM y CatBoost), 25 configuraciones aleatorias, 30 pruebas de Optuna
(TPE) y algoritmo genético con 16 individuos × 10 generaciones. Los espacios están en `src/models.py`.

## 5. Pruebas de hipótesis (α = 0.05)
1. Friedman con prueba posterior de Nemenyi sobre la mejor combinación de cada modelo.
2. t corregida de Nadeau–Bengio y Wilcoxon de rangos con signo, CS-FOF frente a cada modelo,
   con corrección de Holm.
3. DeLong (AUC) y McNemar (errores) sobre predicciones fuera de pliegue, CS-FOF frente a los dos
   mejores modelos base.
4. Tamaño del efecto: delta de Cliff; no despreciable si |δ| ≥ 0.147.

Se reportan p-valores crudos y ajustados. **Superioridad:** se declara solo si se cumplen
simultáneamente las cuatro condiciones de la sección 6.4 del Entregable 3.

## 6. Ablación
Mismas particiones, semillas y presupuesto: (a) sin corte oblicuo; (b) sin costo (c = 1);
(c) solo cortes oblicuos; (d) dirección con todas las variables del nodo.

## 7. Desviaciones respecto a lo planeado
| Planeado | Realizado | Motivo |
|---|---|---|
| Protocolo congelado antes de los experimentos | Escrito después (este documento) | Falta de tiempo; declarado arriba |
| CS-FOF con las 4 técnicas y los 4 métodos | Solo class_weight con malla, aleatoria y Optuna | El genético se estimó en 15 h y las demás técnicas en ~23 h más |
| LightGBM y CatBoost con 16 combinaciones | Solo class_weight sin el genético | Mismo motivo |
| 3 conjuntos de datos adicionales y estudios de robustez | No ejecutados | Mismo motivo |
| Ablación con validación anidada 5 × 5 | Solo en el pliegue 1 (pruebas de diseño) | Mismo motivo |

### Actualización del 9 de noviembre de 2026

Con el plazo extendido al 15 de noviembre se ejecutó parte de lo pendiente. Se registra aquí, sin
modificar la tabla anterior:

| Pendiente | Estado | Nota |
|---|---|---|
| Genético de CS-FOF | Ejecutado | F1 0.5476; no cambia la mejor combinación (aleatoria, 0.5479) |
| LightGBM y CatBoost | class_weight con malla, aleatoria y Optuna | El genético y las otras técnicas quedan en curso; si terminan, se agregan sin cambiar el criterio |
| Conjuntos adicionales | German Credit, Australian Credit Approval y Polish Companies Bankruptcy (solo año 1) | Give Me Some Credit se reemplazó por Polish porque solo está en Kaggle y se pidió un conjunto del repositorio UCI. Se usó un solo año de Polish (7 027 empresas) para que la validación anidada de CS-FOF fuera viable. Los 10 modelos usan class_weight y búsqueda aleatoria de 25 configuraciones (`src/run_additional_datasets.py`) |
| Estudios de robustez | Ruido en etiquetas, tamaño de muestra y calibración (**exploratorios**) | Con los hiperparámetros elegidos en cada pliegue, sin búsqueda nueva (`experiments/robustez/`) |
| Validación externa repetida | No ejecutada | No cambia el veredicto: con F1 casi idéntico al de CatBoost, más repeticiones no producirían superioridad |
| Ablación con validación anidada | Sigue solo en el pliegue 1 | — |
| DeLong y McNemar | Contra CatBoost, XGBoost, LightGBM y Random Forest | Se amplió de los dos mejores modelos base a los cuatro mejores rivales |
