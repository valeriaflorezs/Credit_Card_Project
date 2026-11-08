# Evaluación ampliada: ablación, robustez y limitaciones

## Ablación y sensibilidad (pruebas de diseño)

Las variantes se evaluaron **solo con el entrenamiento del pliegue externo 1**, con validación interna
estratificada de 3 pliegues, sin usar ningún pliegue de prueba (`experiments/cs_fof_diseno/`).

```{list-table} Variantes de diseño de CS-FOF (F1 de la clase *default*, validación interna).
:header-rows: 1
:name: tab-ablacion

* - Variante
  - F1
  - Lectura
* - CS-FOF base (un candidato oblicuo, $q=\sqrt p$, $d=10$, $h=20$)
  - **0.5493**
  - Referencia
* - Sin corte oblicuo (ablación)
  - 0.5452
  - El corte oblicuo aporta +0.004
* - Oblicuo con $p/2$ variables
  - 0.5449
  - Un corte más fuerte sobreajusta
* - Dos candidatos oblicuos por nodo
  - 0.5482
  - Sin mejora
* - Más profundo ($d=14$, $h=10$)
  - 0.5238
  - Sobreajuste claro
* - 128 bins
  - 0.5458
  - Sin mejora
* - Random Forest / XGBoost (mejores hiperparámetros del pliegue 1)
  - 0.5450 / 0.5412
  - Referencias
```

Las variantes de regularización (profundidad 8 a 12, hoja mínima 20 a 40, $\varepsilon$ entre 0.01 y
0.5, `sqrt` o `log2`, 150 o 300 árboles) quedaron todas entre 0.546 y 0.550: el modelo **no depende de
un ajuste fino** de sus hiperparámetros, una señal de estabilidad frente al sobreajuste.

## Estabilidad entre pliegues

En la validación anidada 5 × 5, CS-FOF tiene la menor desviación estándar del F1 entre los tres mejores
modelos (0.0054 frente a 0.0091 de XGBoost y 0.0077 de Random Forest), y sus tres métodos de búsqueda
dan resultados cercanos (0.5456 a 0.5479).

## Lo que no se ejecutó

Por tiempo de cómputo (cada ajuste de CS-FOF tarda unos 14 s en un Ryzen 7 con 8 procesos), no se
ejecutaron: los tres conjuntos de datos adicionales previstos (German Credit, Australian Credit Approval
y Give Me Some Credit), los estudios de ruido, tamaño de muestra y cambio de distribución, la
calibración con recalibración, la predicción conforme y la ablación con validación anidada 5 × 5. Se
declaran como desviaciones en `experiments/protocolo.md` y como trabajo futuro.
