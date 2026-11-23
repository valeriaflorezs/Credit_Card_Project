# Protocolo pre-registrado: stacking con validación cruzada repetida

> **Pre-registro.** Este protocolo se escribe y se versiona **antes** de ejecutar
> `src/run_stacking_repeated.py` con los datos completos. Para dejar constancia, congelarlo con un tag
> de Git antes de la corrida (`git tag protocolo-stacking-v1 && git push origin protocolo-stacking-v1`).
> Cualquier cambio posterior se declara como desviación en la sección 8.

## 1. Motivación
En la validación anidada 5 × 5 (semilla 42), CS-FOF empata estadísticamente con XGBoost y Random
Forest (diferencia de F1 +0.0014, IC 95 % [−0.0059, 0.0086]). El análisis de errores mostró que
XGBoost acierta más entre los clientes **al día** en septiembre (PAY_0 ≤ 0; F1 0.289 frente a 0.261) y
que en los clientes **atrasados** los dos empatan. Un combinador que pondere distinto a cada modelo
según el segmento podría aprovechar esa diferencia. Además, con 5 pliegues la t corregida tiene poca
potencia; la validación repetida 10 × 5 baja la diferencia detectable de unos 0.007 a unos 0.003 de F1.

## 2. Hipótesis
Contrastes **principales** (corrección de Holm sobre los dos):
- **H1.** El stacking por segmento con CS-FOF (`stack_fwls_con`) obtiene un F1 mayor que XGBoost.
- **H2.** `stack_fwls_con` obtiene un F1 mayor que el mismo stacking sin CS-FOF (`stack_fwls_sin`),
  es decir, CS-FOF aporta información que los demás modelos no tienen.

Contrastes **exploratorios** (Holm aparte, no sirven para confirmar):
- cada ensamble (`stack_lr`, `caruana`, `promedio`, en versiones `_con` y `_sin`) y cada modelo base
  frente a XGBoost;
- cada ensamble `_con` frente a su `_sin`.

## 3. Métricas
- **Principal:** F1 de la clase *default* con el umbral que maximiza el F1 en las predicciones
  *out-of-fold* del entrenamiento externo (`f1_umbral`). Se usa el umbral interno, y no 0.5, porque
  los combinadores producen probabilidades en otra escala; XGBoost y los demás modelos base reciben
  el mismo tratamiento.
- **Secundarias:** F1 con umbral 0.5 (`f1_05`), precisión, *recall*, MCC, AUC-ROC y AUC-PR.

## 4. Datos y particiones
- *Default of Credit Card Clients*, 29 965 clientes y 20 variables (`data/processed/dataset_final.csv`).
- **10 repeticiones × 5 pliegues externos estratificados**, semillas **1000 a 1009** (distintas de 42,
  usada en la corrida principal, y de 1 a 4, usadas en `run_repeated_cv`). Las particiones son
  nuevas para todos los modelos.
- 5 pliegues internos para las predicciones *out-of-fold* de nivel 0 y otros 5 para las de nivel 1.

## 5. Modelos
- **Nivel 0:** CS-FOF, XGBoost, CatBoost, LightGBM, Random Forest y regresión logística, todos con
  ponderación de clases y el preprocesamiento del proyecto.
- **Hiperparámetros:** los de `artifacts/metadata*.json` (búsqueda anidada sobre todos los datos).
  **Limitación declarada:** al haberse elegido con todos los datos son algo optimistas, pero lo son
  igual para todos los modelos, así que la comparación es pareada y justa. La versión estricta
  (`--busqueda 25`) se reserva como análisis de sensibilidad.
- **Nivel 1** (`src/stacking.py`), cada uno en versión `_con` y `_sin` CS-FOF:
  - `stack_lr`: regresión logística (pesos balanceados, C = 1) sobre los logits;
  - `stack_fwls`: igual, más las interacciones con `g = 1{PAY_0 > 0}` (*feature-weighted linear
    stacking*, Sill et al., 2009);
  - `caruana`: selección de ensambles con reemplazo (50 pasos, optimiza AUC-PR);
  - `promedio`: promedio simple de probabilidades.

## 6. Pruebas estadísticas (`src/stacking_analysis.py`)
- t corregida de Nadeau y Bengio para validación repetida sobre las 50 diferencias pareadas:
  varianza × (1/50 + n_test/n_train), 49 grados de libertad. α = 0.05 tras Holm.
- t correlacionada bayesiana con ROPE de ±0.01 de F1 (probabilidades de peor, equivalente y mejor).
- Wilcoxon de rangos con signo, solo como referencia (no corrige la dependencia entre pliegues).

## 7. Regla de decisión
- H1 se confirma si p_Holm < 0.05 **y** la diferencia media es positiva.
- H2 se confirma con la misma regla. Si H1 se confirma pero H2 no, se reporta que el ensamble supera
  a XGBoost **sin** evidencia de que CS-FOF sea el responsable.
- Los resultados se reportan sean favorables o no. No se cambia ningún combinador, hiperparámetro ni
  semilla después de ver los pliegues de prueba.

## 8. Desviaciones
- **Prueba de tiempo previa (2026-11-23).** Antes de pre-registrar se corrió un solo pliegue
  (repetición 0, pliegue 0, semilla 1000) con los datos completos para medir el tiempo (unos 111 s por
  pliegue con 4 núcleos). Se detuvo ahí y sus resultados se descartaron. No se cambió nada del
  protocolo después de verlo.

_(Anotar aquí cualquier otro cambio posterior, con fecha y motivo.)_
