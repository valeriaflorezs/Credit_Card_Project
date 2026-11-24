# Protocolo pre-registrado: redes neuronales y comparación final con el stacking

> **Pre-registro.** Se escribe y se versiona **antes** de correr las redes con los datos completos.
> Congelarlo con un tag antes de la corrida:
> `git tag protocolo-redes-v1 && git push origin protocolo-redes-v1`.
> Cualquier cambio posterior se declara en la sección 8.

## 1. Contexto y reparto del trabajo
El curso permite un modelo original basado en redes neuronales con **nota máxima de 4.5**. El equipo
explora dos caminos en paralelo y se queda con el mejor según la regla de la sección 7:

- **Opción A, stacking (sin redes):** `experiments/protocolo_stacking.md`, corrida en
  `results/stacking_repeated/`.
- **Opción B, redes neuronales:** este protocolo, corrida en `results/redes_repeated/`.

Las dos usan **las mismas particiones** (10 repeticiones × 5 pliegues, semillas 1000 a 1009), la
misma métrica y las mismas pruebas, así que sus resultados se pueden parear pliegue a pliegue.

## 2. Modelo original: TabM-Fisher
TabM (Gorishniy et al., ICLR 2025) es un ensamble eficiente de *k* = 32 MLP con pesos compartidos.
**TabM-Fisher** (`src/neural_models.py`) agrega dos ideas del proyecto:

1. **Pérdida sensible al costo:** entropía cruzada con los mismos pesos de CS-FOF
   (w₁ = n₀/n₁ para el moroso, w₀ = 1).
2. **Variables de Fisher:** dentro de cada entrenamiento se ajusta un CS-FOF auxiliar (50 árboles,
   cortes de profundidad ≤ 2). De sus cortes oblicuos (v, t) se toman 16 y cada uno se convierte en la
   distancia con signo vᵀx − t del cliente a esa frontera. La red recibe las 20 variables más esas
   16 distancias.

**Idea:** la red parte de las fronteras diagonales que CS-FOF ya encontró entre atrasos correlacionados
y usa su capacidad en las señales débiles y no lineales de los clientes al día, donde CS-FOF pierde
frente a XGBoost (F1 0.261 frente a 0.289).

## 3. Hipótesis
**Familia final** (Holm sobre las dos; cada opción se compara con el XGBoost de su propia corrida):
- **H-A.** `stack_fwls_con` tiene mayor F1 que XGBoost (es la H1 del protocolo de stacking).
- **H-B.** `tabm_fisher` tiene mayor F1 que XGBoost.

**Secundarias** (Holm aparte):
- **H-B2.** `tabm_fisher` > `tabm` (aporte de las variables de Fisher: ablación).
- `tabpfn` frente a XGBoost y `tabm` frente a XGBoost (referencias neuronales del curso).
- Comparación directa `stack_fwls_con` frente a `tabm_fisher`.

## 4. Métricas
Las del protocolo de stacking: **principal** F1 de la clase *default* con el umbral elegido en las
predicciones *out-of-fold* del entrenamiento (`f1_umbral`); secundarias F1 con umbral 0.5, precisión,
*recall*, MCC, AUC-ROC y AUC-PR.

## 5. Modelos y configuración
| Modelo | Configuración | Búsqueda |
|---|---|---|
| `tabm_fisher` | k = 32, 3 bloques de 512, dropout 0.1, AdamW (lr 2·10⁻³, wd 3·10⁻⁴), lote 256, hasta 100 épocas, paciencia 10 sobre el 10 % del entrenamiento, 16 cortes de Fisher | No (valores por defecto del artículo de TabM) |
| `tabm` | Igual, sin cortes de Fisher | No |
| `tabpfn` | TabPFN v2.5, 8 estimadores, `balance_probabilities=True` | No la necesita (aprende en contexto) |
| `xgboost` | Hiperparámetros de `artifacts/metadata_xgboost.json` | No |

**Pliegues internos:** 3 (y no 5 como en el stacking) para las predicciones *out-of-fold* con que se
elige el umbral, por el costo en GPU (un ajuste de TabM tarda unos 7–8 min en CPU con los datos
completos). Los pliegues **externos** son idénticos a los del stacking, así que el pareo se mantiene.

**Limitación declarada:** XGBoost usa hiperparámetros elegidos con todos los datos y las redes usan los
valores por defecto de sus artículos. Eso favorece a XGBoost, así que una victoria de las redes sería
conservadora. La búsqueda interna (`--busqueda N`) queda como análisis de sensibilidad.

## 6. Pruebas estadísticas
t corregida de Nadeau y Bengio para validación repetida (varianza × (1/J + n_test/n_train), J − 1
grados de libertad), t correlacionada bayesiana con ROPE ±0.01 y Holm por familia
(`src/comparacion_final.py`).

## 7. Regla de decisión (elección entre A y B)
1. Si solo una opción supera a XGBoost (p_Holm < 0.05 y diferencia positiva), se elige esa.
2. Si las dos lo superan y **TabM-Fisher no supera significativamente al stacking**, se elige el
   **stacking**, porque no tiene el tope de 4.5.
3. Si las dos lo superan y TabM-Fisher sí supera al stacking, el equipo decide si la mejora compensa el
   tope de 4.5.
4. Si ninguna supera a XGBoost, se reporta el empate y se mantiene CS-FOF / stacking.

Los resultados se reportan todos, favorables o no. No se cambia ninguna configuración, semilla ni regla
después de ver los pliegues de prueba.

## 8. Desviaciones
- **Prueba de tiempo previa (2026-11-24).** Antes de pre-registrar se ajustaron TabM y TabM-Fisher una
  vez en CPU (repetición 0, pliegue 0) para medir el tiempo. Sus resultados se descartan y no se cambió
  nada del protocolo después de verlos.

_(Anotar aquí cualquier otro cambio posterior, con fecha y motivo.)_
