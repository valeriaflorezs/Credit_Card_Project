# Del CS-FOF al HIB: mejorar el F1 con un ensamble

CS-FOF obtuvo el mayor F1 medio de los 8 modelos (0.5479), pero empató con XGBoost (0.5466). Para mejorar el F1
se partió del modelo original y se hizo un ensamble (*stacking*) con XGBoost, CatBoost, LightGBM, Random Forest,
Extra Trees, HistGradientBoosting y regresión logística. Este capítulo cuenta cómo se llegó al ensamble final,
**HIB** (*Hybrid ensemble with soft-F1 weights*), qué se encontró y qué no se puede afirmar.

:::{admonition} Resumen
:class: cc-limites

- **HIB** combina 8 modelos (CS-FOF y los 7 de arriba) con pesos que se aprenden para maximizar un F1 suave.
- Con semillas nuevas (7100 a 7104, 25 pliegues) obtiene F1 **0.5505** frente a 0.5468 de XGBoost (**+0.0037**, gana en 21 de 25 pliegues) y
  supera a los 7 modelos base con Wilcoxon con Holm, DeLong y McNemar; Friedman global p ≈ $10^{-30}$.
- La ventaja es pequeña: la t corregida de Nadeau–Bengio, que tiene en cuenta que los pliegues comparten datos, no rechaza
  frente a XGBoost ($p = 0.049$ sin ajustar, no sobrevive a Holm).
- La ventaja viene de combinar 8 modelos, no de la regla de pesos: un promedio simple de los mismos modelos logra +0.0004
  menos ($p = 0.33$).
:::

## Qué se hizo

1. **Punto de partida.** CS-FOF y XGBoost empatan; todos los modelos fuertes tienen AUC ≈ 0.785, así que el techo de
   información de los datos es bajo ({numref}`tab-hib-meseta`).
2. **Exploración.** Se probaron promedios de probabilidades, promedios en logit con pesos, pesos por `PAY_0` (*feature-weighted
   linear stacking*) y selección de Caruana {cite:p}`caruana2004` sobre combinaciones de los modelos base: **2 582 combinaciones** en
   10 pliegues de desarrollo.
3. **Congelación.** Se congelaron los 13 mejores candidatos y se validaron con semillas nuevas y 25 pliegues. Todos quedaron entre
   +0.003 y +0.004 de F1 sobre XGBoost: significativo con Wilcoxon con Holm, no con Nadeau–Bengio.
4. **HIB.** Se diseñó una regla propia de pesos y se evaluó en dos corridas pre-registradas (v1: semillas 5000 a 5004; v2:
   semillas 7100 a 7104), cada una de 5 repeticiones × 5 pliegues.

Los datos son los mismos 29 965 clientes de Taiwán, con 45 variables: las 20 del conjunto final y 25 derivadas de facturas y
pagos (utilización del cupo, razón pago/factura, meses con pago mínimo, racha de atraso, entre otras;
`experiments/hib/gen_taiwan.py` y `conf_lib.py`). Los 7 modelos base de comparación son XGBoost, Random Forest, SVM, KNN, regresión
logística, árbol de decisión y Naive Bayes, con los hiperparámetros del repositorio y el mismo umbral elegido en el bucle interno.

## La regla de HIB

HIB toma los logits fuera de muestra $Z$ de sus $M = 8$ miembros (CS-FOF, XGBoost, CatBoost, LightGBM, Random Forest, Extra Trees,
HistGradientBoosting y regresión logística), calculados con 3 pliegues internos dentro del entrenamiento de cada pliegue externo, y aprende
cuánto pesa cada uno:

$$
w=\operatorname{softmax}(\theta),\qquad s=Zw
$$ (eq-hib-pesos)

$$
q_i=\sigma\!\left(\frac{s_i-t}{\tau}\right)
$$ (eq-hib-q)

$$
\max_{\theta,\,t}\ \frac{2\sum_i q_i y_i}{\sum_i q_i+\sum_i y_i}\;-\;\lambda M\,\lVert w-\mathbf{1}/M\rVert^{2}
$$ (eq-hib-objetivo)

- {eq}`eq-hib-pesos`: el *softmax* mantiene los pesos positivos y con suma 1; $s$ es el promedio ponderado de los logits.
- {eq}`eq-hib-q`: marcar o no a un cliente como incumplido es un escalón y no se puede derivar; $q_i$ es su versión suave, cerca de 1 si
  $s_i$ supera el umbral $t$ y cerca de 0 si no. Con $\tau = 0.3$ la curva se parece a un escalón pero sigue siendo derivable.
- {eq}`eq-hib-objetivo`: el primer término es el F1 suave, como en {cite:t}`benedict2021` y {cite:t}`lee2021`, y se maximiza con
  L-BFGS-B (60 iteraciones). El segundo, con $\lambda = 0.05$, empuja los pesos hacia el promedio uniforme $1/M$; por eso los pesos
  aprendidos quedan casi iguales (entre 0.10 y 0.14).
- **Umbral final.** No es el $t$ aprendido: se elige el que maximiza el F1 duro sobre las puntuaciones fuera de muestra (entre 5 % y
  60 % de positivos predichos), como en {cite:t}`lipton2014` y como para todos los modelos.

El código es la función `hib_scores` (`experiments/hib/hib.py`):

```python
import numpy as np
from scipy.optimize import minimize
from combos import logit, thr_fast, f1_at
def hib_scores(O, y, T, lam=0.05, tau=0.3):
    """HIB: pesos softmax sobre logits que maximizan un F1 suave (regla propia, protocolo congelado)."""
    Z, Zt = logit(O), logit(T); m = Z.shape[1]; t0 = thr_fast(y, Z.mean(1))
    def f(p):
        w = np.exp(p[:m]-p[:m].max()); w /= w.sum(); q = 1/(1+np.exp(-(Z @ w-p[m])/tau))
        return -(2*(q*y).sum()/(q.sum()+y.sum())) + lam*m*((w-1/m)**2).sum()
    r = minimize(f, np.r_[np.zeros(m), t0], method="L-BFGS-B", options=dict(maxiter=60)); w = np.exp(r.x[:m]-r.x[:m].max()); w /= w.sum()
    return Z @ w, Zt @ w, w
```

`logit`, `thr_fast` y `f1_at` son auxiliares del entorno de las corridas (`combos`, `common2`) y no están en el repositorio; los miembros vienen de
librería y solo CS-FOF es propio.

## Los candidatos congelados

{numref}`tab-hib-candidatos` muestra los candidatos congelados que no usan *stacking* con regresión logística como combinador. Diferencia =
candidato − XGBoost, con 25 pliegues y semillas nuevas. El valor de Wilcoxon es el ajustado por Holm.

```{list-table} Candidatos congelados frente a XGBoost (fase B, 25 pliegues).
:header-rows: 1
:name: tab-hib-candidatos

* - Candidato
  - F1
  - Dif. F1 [IC 95 %]
  - Gana en
  - p Nadeau–Bengio
  - p Wilcoxon (Holm)
* - Promedio: CS-FOF + XGBoost (prof. 4) + CatBoost + CatBoost (prof. 8) + HistGradientBoosting
  - 0.5499
  - +0.0038 [−0.0020, +0.0096]
  - 19/25
  - 0.19
  - 0.0127
* - Logit ponderado: CatBoost + HistGradientBoosting
  - 0.5495
  - +0.0034 [−0.0032, +0.0100]
  - 18/25
  - 0.29
  - 0.0295
* - Promedio: XGBoost (prof. 4) + CatBoost + CatBoost (prof. 8) + LightGBM + HistGradientBoosting
  - 0.5494
  - +0.0032 [−0.0028, +0.0092]
  - 19/25
  - 0.28
  - 0.0295
* - Logit ponderado: XGBoost (prof. 4) + CatBoost + HistGradientBoosting
  - 0.5492
  - +0.0031 [−0.0029, +0.0091]
  - 17/25
  - 0.30
  - 0.0295
* - FWLS por `PAY_0`: XGBoost (prof. 4) + CatBoost
  - 0.5491
  - +0.0030 [−0.0030, +0.0090]
  - 16/25
  - 0.31
  - 0.0295
* - Logit ponderado: CatBoost + CatBoost (prof. 8) + LightGBM + HistGradientBoosting
  - 0.5491
  - +0.0030 [−0.0022, +0.0082]
  - 19/25
  - 0.25
  - 0.0135
* - Selección de Caruana, 10 modelos (con CS-FOF)
  - 0.5491
  - +0.0030 [−0.0014, +0.0074]
  - 20/25
  - 0.17
  - 0.0117
* - FWLS por `PAY_0`, 10 modelos (con CS-FOF)
  - 0.5491
  - +0.0030 [−0.0017, +0.0078]
  - 15/25
  - 0.20
  - 0.0209
```

Los 8 pasan Wilcoxon con Holm, pero ninguno pasa Nadeau–Bengio. Por eso se diseñó HIB y se confirmó con un protocolo pre-registrado. El
resto de la exploración (codificación WoE de `PAY_x`, variables derivadas de `PAY`, LightGBM con árboles lineales, Rotation Forest propio y un piloto
de 6 componentes complementarios) tampoco superó a XGBoost con significancia. La tabla completa está en `experiments/hib/Tabla_combinaciones.xlsx`.

## Protocolo y confirmación

- **v1** (semillas 5000 a 5004). Pruebas fijadas: t de Nadeau–Bengio, Wilcoxon y Nemenyi, con Holm sobre 7 contrastes
  (`experiments/hib/PROTOCOLO_CONFIRMATORIO.md`, con su hash SHA-256).
- **v2** (semillas 7100 a 7104). Tras ver v1 se exploraron McNemar y DeLong sobre las mismas predicciones, así que esa elección de pruebas se hizo
  mirando datos. Por eso v2 es una confirmación independiente, con las pruebas fijadas antes de generar los datos: Wilcoxon de rangos con signo
  (aceptada por el docente) sobre las 25 diferencias de F1 por pliegue, DeLong pareado sobre el AUC y McNemar, cada una con Holm sobre 7
  contrastes, más Friedman global; Nadeau–Bengio y Nemenyi se reportan sin que decidan
  (`experiments/hib/PROTOCOLO_CONFIRMATORIO_v2.md`; resultados en `conf2_tw2_*`).
- **Criterio de superioridad frente a un modelo base:** F1 medio de HIB mayor; Wilcoxon, DeLong y McNemar con p ajustado < 0.05;
  $|\delta_{\text{Cliff}}| \ge 0.147$; y Friedman global significativo.

```{list-table} Confirmación v2 (semillas 7100 a 7104, 25 pliegues). Diferencia = HIB − modelo; p ajustados por Holm sobre 7 contrastes.
:header-rows: 1
:name: tab-hib-v2

* - Modelo
  - F1 medio
  - Dif. F1
  - Pliegues ganados por HIB
  - $\delta$ de Cliff
  - Wilcoxon
  - DeLong (AUC)
  - McNemar
* - **HIB**
  - **0.5505**
  - —
  - —
  - —
  - —
  - —
  - —
* - Random Forest
  - 0.5477
  - +0.0028
  - 17/25
  - 0.24
  - 0.0147
  - < 0.0001
  - 0.001
* - XGBoost
  - 0.5468
  - +0.0037
  - 21/25
  - 0.29
  - 0.0001
  - 0.0003
  - < 0.0001
* - SVM
  - 0.5411
  - +0.0094
  - 25/25
  - 0.62
  - < 0.0001
  - < 0.0001
  - < 0.0001
* - Árbol de decisión
  - 0.5326
  - +0.0179
  - 25/25
  - 0.84
  - < 0.0001
  - < 0.0001
  - 0.001
* - Regresión logística
  - 0.5266
  - +0.0239
  - 25/25
  - 0.97
  - < 0.0001
  - < 0.0001
  - < 0.0001
* - KNN
  - 0.5233
  - +0.0271
  - 25/25
  - 0.98
  - < 0.0001
  - < 0.0001
  - < 0.0001
* - Naive Bayes
  - 0.5106
  - +0.0399
  - 25/25
  - 1.00
  - < 0.0001
  - < 0.0001
  - < 0.0001
```

Friedman global: $p \approx 10^{-30}$; rango medio de HIB 1.48 frente a 2.28 de Random Forest y 2.52 de XGBoost. Las pruebas que **no** rechazan frente
a XGBoost y Random Forest son Nadeau–Bengio ($p$ sin ajustar 0.049 y 0.31) y Nemenyi (0.81 y 0.94); en v2 eran informativas, no decisivas.

La **primera corrida (v1)** es consistente: HIB 0.5502, XGBoost 0.5463 y Random Forest 0.5471; diferencias +0.0039 y +0.0031; Wilcoxon con Holm 0.0016
en ambos casos; Nadeau–Bengio 0.17 y 0.16 sin ajustar; 19 de 25 pliegues ganados.

## De dónde sale la ventaja

Comparación exploratoria (no pre-registrada, hecha después de ver v2) de la regla de HIB contra reglas estándar sobre las mismas predicciones de v2:

```{list-table} HIB frente a reglas estándar con los mismos 8 modelos.
:header-rows: 1
:name: tab-hib-reglas

* - Regla con los mismos 8 modelos
  - F1 medio
  - HIB − regla
  - Pliegues ganados por HIB
  - Wilcoxon
* - **HIB (F1 suave)**
  - **0.5505**
  - —
  - —
  - —
* - Promedio uniforme de logits
  - 0.5501
  - +0.0004
  - 15/25
  - 0.33
* - Promedio de probabilidades
  - 0.5501
  - +0.0004
  - 16/25
  - 0.30
* - Selección de Caruana
  - 0.5494
  - +0.0011
  - 14/25
  - 0.12
```

Un promedio simple de los mismos 8 modelos también supera a XGBoost y a Random Forest; la regla de pesos de HIB no lo mejora de forma medible. Quitar CS-FOF
tampoco cambia nada (0.5505 sin él frente a 0.5502 con él en v1). Los miembros que mueven el resultado son los de *boosting*.

## Por qué no mejora más

```{list-table} F1 medio de cada modelo en las corridas (meseta).
:header-rows: 1
:name: tab-hib-meseta

* - Modelo
  - F1 medio
* - Regresión logística
  - 0.5230
* - Rotation Forest propio
  - 0.5421
* - XGBoost
  - 0.5457
* - Random Forest
  - 0.5461
* - LightGBM
  - 0.5467
* - CS-FOF
  - 0.5473
* - CatBoost
  - 0.5494
* - Promedio de CS-FOF y 4 modelos de *boosting*
  - 0.5499
* - **HIB (v2)**
  - **0.5505**
```

- **Las diferencias son del tamaño del ruido.** El F1 de un mismo modelo varía alrededor de 0.01 de un pliegue a otro (desviación estándar de 0.009 a 0.012),
  mientras que las diferencias entre los mejores modelos son de 0.002 a 0.004. La t corregida de Nadeau–Bengio multiplica la varianza por
  $1/J + n_{\text{prueba}}/n_{\text{entrenamiento}} = 1/J + 0.25$; el segundo término no baja al agregar repeticiones, porque los pliegues comparten la
  mayor parte del entrenamiento. Pasar de 25 a 50 pliegues casi no cambia el umbral de significancia.
- **El límite está en los datos.** Árboles por ejes, cortes oblicuos de Fisher, *boosting*, ensambles y HIB convergen al mismo F1 ≈ 0.54 a 0.55 y AUC ≈ 0.78 a 0.79,
  lo mismo que reporta la literatura sobre este conjunto. La señal dominante es el atraso reciente (`PAY_0` y `PAY_2`); en los clientes al día en septiembre
  (77 % de la cartera, 13.8 % de incumplimiento) ningún modelo pasa de F1 0.29.
- **Los modelos se equivocan en los mismos clientes.** Por eso sumar 8 modelos da una ganancia pequeña y la forma de ponderarlos casi no importa.
- **Ordenar mejor no es clasificar mejor.** Una mejora del AUC reordena pocos clientes cerca del umbral y el F1 casi no cambia.

Con estos datos, una mejora sustancial del score es muy improbable: la única vía realista sería agregar información nueva (otras variables u otros periodos).

## Lo que no se puede afirmar

- **Tamaño del efecto.** +0.0037 de F1 (≈ 0.7 % relativo) puede considerarse marginal en un conjunto con techo de AUC ≈ 0.785.
- **Dependencia entre pliegues.** Wilcoxon trata los 25 pliegues como independientes, y DeLong y McNemar se aplican sobre predicciones agrupadas; las tres ignoran que
  los pliegues comparten datos y dan p optimistas. La prueba que sí lo controla (Nadeau–Bengio) no rechaza frente a XGBoost ni Random Forest. Incluso CatBoost solo supera a
  XGBoost con Wilcoxon con Holm en la exploración (p = 0.0093).
- **Exploración previa.** Se evaluaron más de 2 600 diseños antes de HIB; las semillas nuevas en cada etapa lo mitigan, pero la ventaja observada del mejor de muchos tiende a estar inflada.
- **La regla y CS-FOF no aportan de forma medible** frente al promedio uniforme y frente a HIB sin CS-FOF; la novedad defendible es la combinación aplicada a este problema, con precedentes en
  la optimización del ensamble para una métrica {cite:p}`caruana2004` y en la sustitución suave del F1 {cite:p}`benedict2021,lee2021`.
- **Un solo conjunto de datos y particiones distintas por corrida.** Solo se evaluó en Taiwán (2005); cada corrida se compara con el XGBoost de su propia partición.
- **Trazabilidad.** Los protocolos v1 y v2 llevan su hash SHA-256 y se registraron antes de generar los datos de cada corrida. El *tag* de Git no se pudo subir a GitHub (error 403) y existe
  solo en local, y después los protocolos se editaron para quitar los conjuntos adicionales, con los hashes antes y después registrados.
- **Presupuesto.** Los modelos base y los miembros usan los hiperparámetros del repositorio, sin búsqueda nueva, y no se reportó el cómputo; HIB entrena 4 veces cada miembro (3 pliegues internos y el ajuste final).

## Archivos

Todo está en `experiments/hib/`: la regla (`hib.py`), la generación de logits (`gen_taiwan.py`, `conf_lib.py`), el análisis (`analyze_conf2.py`), los protocolos con sus hashes,
los resultados de la confirmación v2 (`conf2_tw2_result.txt`, `conf2_tw2_contrastes.csv`, `conf2_tw2_f1.csv`) y la tabla de combinaciones (`Tabla_combinaciones.xlsx`). El tablero lo cuenta en
la pestaña «Del CS-FOF al HIB» de la página Modelo original.
