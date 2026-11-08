# Declaración de los datos

## Procedencia
- **Conjunto:** *Default of Credit Card Clients*, UCI Machine Learning Repository.
  https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- **Publicación original:** Yeh, I. C. y Lien, C. H. (2009). The comparisons of data mining techniques
  for the predictive accuracy of probability of default of credit card clients. *Expert Systems with
  Applications*, 36(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020
- **Licencia:** Creative Commons Attribution 4.0 International (CC BY 4.0), según el repositorio UCI.
- **Cómo obtenerlo:** descargar el `.xls` desde UCI y guardarlo como
  `data/raw/default_of_credit_card_clients.xls`. No se versiona en el repositorio (`.gitignore`).

## Contenido
- 30 000 clientes de un banco de Taiwán; variables de abril a septiembre de 2005; la variable objetivo
  indica si el cliente incumplió el pago de octubre de 2005 (22.1 % sí).
- 23 variables predictoras: límite de crédito, sexo, educación, estado civil, edad, seis estados de
  pago mensuales, seis montos facturados y seis montos pagados.
- No hay datos personales identificables: el `ID` se descarta y no hay nombres ni documentos.

## Procesamiento (`src/preprocessing.py`)
1. Eliminación de 35 filas duplicadas → 29 965 clientes.
2. Códigos no documentados de `EDUCATION` (0, 5, 6) y `MARRIAGE` (0) reagrupados en «Otros».
3. Logaritmo con signo de `LIMIT_BAL` y `PAY_AMT1`–`PAY_AMT6`.
4. El bloque `BILL_AMT1`–`BILL_AMT6` (VIF de hasta 25) se reemplaza por nivel medio, tendencia y
   número de meses con saldo a favor → 20 variables.
5. La codificación one-hot, el escalado y el remuestreo se ajustan dentro de cada pliegue, sin fuga.

## Sesgos y limitaciones conocidos
- Un solo banco, un solo país y un periodo de seis meses de 2005, en medio de una crisis de deuda
  de tarjetas en Taiwán: no es representativo de otros contextos.
- Los códigos −2 y 0 de las variables `PAY_x` no están bien documentados por la fuente.
- Hay 21 grupos de clientes con variables idénticas y etiquetas distintas: ruido irreducible.
- Las variables `SEX`, `EDUCATION`, `MARRIAGE` y `AGE` son atributos sensibles; las tasas de
  incumplimiento varían entre grupos, lo que puede introducir sesgo en los modelos.
