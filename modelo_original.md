# Modelo original: Cost-Sensitive Fisher Oblique Forest (CS-FOF)

## Idea central

Un árbol de decisión convencional solo pregunta por una variable a la vez. Cuando la frontera entre
clientes que pagan y clientes que incumplen es diagonal entre variables correlacionadas, como los
estados de pago de meses consecutivos, el árbol la aproxima con una escalera de cortes que fragmenta
a la clase minoritaria. **CS-FOF** es un bosque aleatorio en el que cada nodo puede cortar también por
una **combinación lineal de variables**: la dirección discriminante de Fisher {cite:p}`fisher1936`,
ponderada por el costo del incumplimiento y calculada solo con los clientes que llegan a ese nodo.
El nodo compara ese corte con los cortes por ejes y conserva el mejor.

## Formulación

Sea $\mathcal{D}=\{(\mathbf{x}_i,y_i)\}_{i=1}^n$, con $\mathbf{x}_i\in\mathbb{R}^p$ y $y_i\in\{0,1\}$
($1$ = incumple). Cada cliente recibe un peso de costo

$$
w_i = c\,\mathbb{1}\{y_i=1\} + \mathbb{1}\{y_i=0\}, \qquad c=\frac{n_0}{n_1}.
$$ (eq-pesos)

Un corte que divide los clientes de un nodo en $L$ y $R$ se evalúa con la impureza de Gini ponderada

$$
G(L,R)=W_L\,2p_L(1-p_L)+W_R\,2p_R(1-p_R),\qquad W_A=\sum_{i\in A}w_i,\quad p_A=\frac{\sum_{i\in A}w_iy_i}{W_A}.
$$ (eq-gini)

**Candidato oblicuo.** Con las $q$ variables sorteadas en el nodo, $F$, se calculan las medias por clase
$\boldsymbol\mu_0,\boldsymbol\mu_1$ y la dispersión intra-clase ponderada y regularizada

$$
S_w=\frac{1}{\sum_i w_i}\sum_{i}w_i(\mathbf{x}_{i,F}-\boldsymbol\mu_{y_i})(\mathbf{x}_{i,F}-\boldsymbol\mu_{y_i})^\top+\varepsilon\,\frac{\operatorname{tr}S}{q}\,I_q,
$$ (eq-sw)

y la dirección del corte es

$$
\mathbf{v}=\frac{S_w^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_0)}{\lVert S_w^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_0)\rVert}.
$$ (eq-fisher)

El umbral se busca sobre las proyecciones $z_i=\mathbf{v}^\top\mathbf{x}_{i,F}$ minimizando
{eq}`eq-gini`. La probabilidad final es el promedio de las $T$ probabilidades de hoja ponderadas:

$$
f(\mathbf{x})=\frac1T\sum_{t=1}^T\hat p_t(\mathbf{x}),\qquad \hat p_{\text{hoja}}=\frac{\sum_{i\in\text{hoja}}w_iy_i}{\sum_{i\in\text{hoja}}w_i}.
$$ (eq-prediccion)

## Algoritmo

1. Calcular los pesos $w_i$ con {eq}`eq-pesos`.
2. Para cada árbol $t=1,\dots,T$: tomar una muestra *bootstrap* y crecer el árbol **por niveles**.
3. En cada nodo, si no se alcanzó la profundidad $d$ ni el mínimo de $2h$ clientes y el nodo no es puro:
   1. sortear $q=\lfloor\sqrt p\rfloor$ variables y buscar el mejor umbral por eje con {eq}`eq-gini`;
   2. calcular $\mathbf v$ con {eq}`eq-sw`–{eq}`eq-fisher` y el mejor umbral sobre la proyección;
   3. conservar el corte de menor $G$ que deje al menos $h$ clientes por lado; si no hay, hoja.
4. Predecir con {eq}`eq-prediccion`.

Los umbrales se buscan con histogramas de 64 cuantiles, como en LightGBM o el modo `hist` de XGBoost.

## Propiedades

**Proposición 1.** La impureza del corte elegido nunca supera la del mejor corte por ejes sobre las
mismas variables. *Demostración:* el conjunto de candidatos contiene los cortes por ejes, y el mínimo
sobre un conjunto es menor o igual que sobre cualquier subconjunto. ∎

**Proposición 2.** Para $\varepsilon=0$ y $S_w$ definida positiva, $\mathbf v$ maximiza
$J(\mathbf u)=(\mathbf u^\top(\boldsymbol\mu_1-\boldsymbol\mu_0))^2/\mathbf u^\top S_w\mathbf u$.
*Demostración:* por Cauchy–Schwarz con el producto interno $\langle a,b\rangle=a^\top S_w b$,
$(\mathbf u^\top\mathbf d)^2\le(\mathbf u^\top S_w\mathbf u)(\mathbf d^\top S_w^{-1}\mathbf d)$, con
igualdad si y solo si $\mathbf u\propto S_w^{-1}\mathbf d$. ∎

**Proposición 3.** Cada árbol termina y tiene a lo sumo $\lfloor n/h\rfloor$ hojas, porque solo se
aceptan cortes con al menos $h$ clientes por lado y la profundidad está acotada. ∎

**Complejidad.** Un árbol cuesta $O(d\,n(q\log n+q^2))$ en entrenamiento y la predicción de un cliente
$O(T\,d\,p)$. Las tres proposiciones se verifican numéricamente en `tests/test_original_model.py`.

## Implementación

Clase `CSFisherObliqueForest` en `src/original_model.py`, escrita con numpy y compatible con
scikit-learn (`fit`, `predict`, `predict_proba`). Los árboles se reparten en procesos (`n_jobs`) con
semillas propias, de modo que el resultado no depende del número de procesos. Las 14 pruebas unitarias
cubren aprendizaje, reproducibilidad, entradas inválidas, `clone`, *pipelines* y las propiedades, y se
ejecutan en cada *push* con GitHub Actions.

## Qué es nuevo y qué se reutiliza

Se reutilizan la estructura del bosque aleatorio {cite:p}`breiman2001` y la idea de cortes oblicuos
{cite:p}`murthy1994,menze2011`. Lo nuevo es la **regla de partición**: una dirección discriminante local,
ponderada por costo, que compite con los cortes por ejes bajo el mismo criterio ponderado.

## Iteraciones de diseño

Se evaluaron 14 variantes, **solo con el entrenamiento del pliegue externo 1** (validación interna de
3 pliegues), sin tocar los datos de prueba (`experiments/cs_fof_diseno/`). Las versiones con cortes
oblicuos más fuertes (más variables o árboles más profundos) **sobreajustaron**; por eso se fijaron un
candidato oblicuo por nodo con las mismas $q$ variables y una profundidad máxima de 14.
