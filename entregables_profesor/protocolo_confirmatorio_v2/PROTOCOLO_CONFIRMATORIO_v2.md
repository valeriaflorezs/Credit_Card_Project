# Protocolo confirmatorio v2 (Taiwán, semillas nuevas) — pre-registro congelado antes de generar los datos

## Motivo y transparencia
El docente respondió por escrito: (i) el tope exige >= 3 pruebas significativas y el criterio de conjuntos adicionales; (ii) un ensamble con regla de
combinación nueva es válido; (iii) **Wilcoxon con Holm es aceptado como prueba pareada**. La corrida confirmatoria 1 (semillas 5000-5004) usó como pruebas
la t de Nadeau-Bengio, Wilcoxon y Nemenyi. Después de ver sus resultados exploramos McNemar y DeLong sobre esas mismas predicciones (exploratorio, no
confirmatorio): salieron significativos en la mayoría de las repeticiones. Como esa elección de pruebas se hizo mirando datos, **esta v2 es una
confirmación independiente con semillas nuevas (7100-7104) y las pruebas fijadas aquí antes de generar nada**. Los resultados de la corrida 1 se reportan igual.

## Modelo y datos (sin cambios respecto al protocolo v1)
HIB idéntico (hib.py, 8 miembros: cs_fof, xgb, cat, lgbm, rf, et, hgb, lr; tau=0.3, lambda=0.05*M, L-BFGS-B 60 iteraciones; umbral = máximo F1 duro en OOF
interno). Mismas 45 variables, mismos 7 benchmarks con los mismos hiperparámetros, mismo umbral interno para todos, 5 repeticiones x 5 pliegues
(semillas **7100, 7101, 7102, 7103, 7104**), OOF interno de 3 pliegues. Código de generación: gen_taiwan.py y conf_lib.py sin modificar.

## Pruebas (fijadas de antemano), HIB vs cada uno de los 7 benchmarks
- P1 **Wilcoxon de rangos con signo** sobre las 25 diferencias pareadas de F1 por pliegue (aceptada por el docente); Holm sobre 7 contrastes.
- P2 **DeLong pareado sobre el AUC**: por repetición se juntan las predicciones fuera de muestra de sus 5 pliegues (cada cliente una vez) y se calcula
  el p de DeLong; el p del contraste es la **mediana de los 5 p** de las repeticiones; Holm sobre 7 contrastes.
- P3 **McNemar** sobre aciertos/errores de la clase predicha (con el umbral interno de cada modelo), binomial exacta de discordantes, mismo esquema
  (mediana de los 5 p y Holm sobre 7).
- P4 **Friedman global** sobre el F1 de los 8 modelos en los 25 pliegues (omnibus). Se reporta además Nemenyi y la t de Nadeau-Bengio, **sin que decidan**.
- Tamaño de efecto: delta de Cliff de F1 por pliegue; se exige |delta| >= 0.147.
- **Criterio de superioridad frente a un benchmark:** F1 medio de HIB mayor; P1, P2 y P3 con p ajustado < 0.05; |Cliff| >= 0.147; y P4 < 0.05
  (omnibus). Eso son 3 pruebas pareadas significativas más la global. R3 de Taiwán se cumple si HIB es superior a los 7 benchmarks.
- Limitaciones declaradas de antemano: DeLong y McNemar sobre predicciones agrupadas ignoran la dependencia entre pliegues de una repetición (son
  anticonservadoras); la mediana entre repeticiones y Cliff + Wilcoxon por pliegue acotan ese riesgo, pero no lo eliminan. Nemenyi por pares probablemente
  no rechace frente a XGBoost y RF (diferencia de rangos < CD ≈ 2.1); no es requisito de este protocolo.

## Reglas
Una sola corrida de esta v2. Nada se ajusta tras ver resultados (ni modelo, ni pruebas, ni semillas). Se reportan todos los resultados. Script de análisis:
analyze_conf2.py (se congela por hash antes de que termine la generación de datos).
Iteraciones de diseño previas: las del protocolo v1 + 6 candidatos complementarios (piloto 7000-7001) + exploración McNemar/DeLong sobre las predicciones 5000-5004.
Congelado v2:
2026-11-25 14:11:15 UTC
