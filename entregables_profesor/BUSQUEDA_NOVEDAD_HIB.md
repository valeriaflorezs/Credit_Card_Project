# Búsqueda de novedad de HIB (ensamble con pesos por F1 suave) — default de tarjetas de crédito

Fecha de la búsqueda: 2026-11-25. Método: motor de búsqueda web general (no consultas directas a IEEE Xplore, Scopus ni ACM). Se revisaron los
títulos, resúmenes y fragmentos devueltos, no los textos completos. **Es una búsqueda preliminar, no la revisión sistemática de la Sección 5.2 de la guía.**

## Método evaluado (HIB)
Pesos w = softmax(θ) sobre los logits fuera de muestra de M modelos base; θ y un umbral t se optimizan (L-BFGS-B) para maximizar un F1 suave
2·Σ(q·y)/(Σq + Σy), con q = sigmoide((s − t)/τ), s = Z·w, τ = 0.3; penalización λ·M·‖w − 1/M‖² (λ = 0.05); umbral final = máximo F1 duro sobre las
puntuaciones fuera de muestra. Evaluado por F1 de la clase default en UCI Taiwán.

## Matriz de novedad (11 cadenas; 2026-11-25)
| # | Cadena (resumida) | Idioma | Resultado relevante |
|---|---|---|---|
| 1 | ensemble weights optimized for F1, soft F1 surrogate, stacking, imbalanced | EN | Lee et al. 2021 (F-beta surrogate); Evo* 2025 (pesos por evolución diferencial sobre macro-F1) |
| 2 | ensemble selection Caruana, arbitrary metric F1, hillclimbing | EN | Caruana et al. 2004; Munson, Cardie, Caruana 2005; Weka y pyensemble permiten F1 como métrica |
| 3 | credit card default Taiwan UCI stacking ensemble weighted F1 | EN | Varios stackings sobre Taiwán; ninguno con pesos optimizados para F1 |
| 4 | soft F1 loss, sigmoid approximation, Lipton, Dinkelbach | EN | sigmoidF1 (Bénédict et al. 2021); Lipton et al. 2014 (umbral óptimo) |
| 5 | ensamble ponderado, F1, incumplimiento de tarjetas de crédito | ES | Stacking de fraude (PeerJ 2025); BlendNet (2026); sin coincidencia exacta |
| 6 | optimizing ensemble weights for F-measure, softmax weights | EN | Cui et al. (UAI 2026): pesos de ensamble + F1 suave + L2 |
| 7 | "Ensemble Diversity Optimization for Subjective Supervision" (verificación) | EN | Confirmado en arXiv y PMLR |
| 8 | "Optimised Greedy-Weighted Ensemble ... Loan Default" (verificación) | EN | Confirmado en arXiv (2026), Lending Club |
| 9 | F1-optimized stacking meta-learner, threshold, credit default | EN | Sin coincidencia exacta; stacking con umbral en crédito (MDPI 2026) |
| 10 | Taiwan credit card default blending, evolución diferencial/genético, F1 | EN | Comparaciones de ensambles en Taiwán (UHD 2021); sin pesos optimizados para F1 |
| 11 | Lipton, Elkan, Narayanaswamy, umbral óptimo F1 (verificación) | EN | Confirmado (arXiv 1402.1892; ECML 2014) |

## Trabajos más cercanos (a verificar en su texto completo)
1. **Cui, Huang, Abeynayake (2026). Ensemble Diversity Optimization for Subjective Supervision.** UAI 2026 (PMLR 337); arXiv 2607.08493. Aprende pesos de ensamble
   con una pérdida diferenciable que incluye un **micro-F1 suave** y una penalización L2 sobre los pesos (más calibración y diversidad). Es **texto con anotadores,
   no crédito**. Es el precedente más cercano del *mecanismo*.
2. **Caruana, Niculescu-Mizil, Crew, Ksikes (2004). Ensemble Selection from Libraries of Models.** ICML 2004. Selección hacia adelante que **maximiza cualquier
   métrica** (incluido F1 en las implementaciones de Weka y pyensemble). Es la regla estándar contra la que HIB debe contrastarse.
3. **Munson, Cardie, Caruana (2005).** Ensemble selection optimizando métricas de PLN (HLT-EMNLP).
4. **Evo\* 2025, resumen tardío (arXiv 2511.17543).** Evolución diferencial para pesos de ensamble con **macro-F1** en TASS 2019 (sentimiento en español).
5. **Lee et al. (2021). A surrogate loss function for optimization of F-beta score** (arXiv 2104.01459) y **Bénédict et al. (2021) sigmoidF1** (arXiv 2108.10566):
   sustitutos diferenciables del F1 (modelo individual).
6. **Lipton, Elkan, Narayanaswamy (2014). Optimal Thresholding of Classifiers to Maximize F1** (ECML PKDD; arXiv 1402.1892): relación entre el umbral y el F1 óptimo.
7. **En crédito:** Nortey et al. (2026), *An Optimised Greedy-Weighted Ensemble Framework for Financial Loan Default Prediction* (arXiv 2603.18927; ponderación greedy
   y meta-aprendiz neuronal, Lending Club); stackings sobre Taiwán (p. ej. UHD J. Sci. Technol. 2021; tesis NCCU); TabNet-Stacking (PMC11506879); STRIKE (arXiv 2604.17622).

## Tabla comparativa por familia (qué se conserva, qué cambia)
| Familia | Trabajo | Qué comparte con HIB | Qué lo distingue de HIB |
|---|---|---|---|
| Selección de ensambles | Caruana 2004 | Optimiza pesos para una métrica (F1 posible) | Búsqueda greedy discreta, sin penalización a pesos uniformes |
| Pesos con F1 suave | Cui et al. 2026 | Pesos aprendidos con F1 suave + L2 | Texto con anotadores; mezcla CE, diversidad y calibración |
| Pesos evolutivos | Evo\* 2025 | Pesos que maximizan F1 (macro) | Evolución diferencial, sin gradiente; PLN |
| Sustitutos de F1 | Lee 2021; sigmoidF1 | Aproximación suave del F1 | Entrenan un único modelo, no pesos de ensamble |
| Umbral óptimo | Lipton 2014 | Umbral que maximiza F1 | No combina modelos |
| Stacking en crédito | Nortey 2026; stackings Taiwán | Combina modelos de crédito | Meta-aprendiz neuronal o logístico; no optimiza F1 directo |

## Conclusión provisional (sin afirmar novedad)
- **No encontré, en esta búsqueda preliminar, un trabajo que combine** exactamente (a) pesos softmax sobre logits, (b) F1 suave con τ fijo y penalización a pesos
  uniformes, y (c) el problema de default de tarjetas (UCI Taiwán). Eso es novedad de *aplicación y de combinación de piezas*.
- **El mecanismo central no es nuevo:** optimizar los pesos de un ensamble para F1 aparece en Caruana 2004 (con F1 posible), en Evo\* 2025 y, con F1 suave
  diferenciable y L2, en Cui et al. 2026. La guía (Cuadro 1) rechaza la novedad "solo de aplicación, sin mecanismo algorítmico nuevo" y exige una regla de
  ponderación "con regla de actualización derivada y contrastada contra reglas estándar".
- **Riesgo alto para R1.** Para defenderla habría que mostrar una contribución algorítmica propia (por ejemplo, una regla de actualización derivada, una
  propiedad demostrada o una versión por región o por incertidumbre) y contrastarla con Caruana, promedio simple y stacking lineal.

## Lo que no pude verificar
- No consulté IEEE Xplore, Scopus ni ACM directamente, y no leí textos completos. La búsqueda en español no devolvió trabajos relevantes sobre pesos optimizados para F1 en default.
- Tran et al. (2022), *An Ensemble Learning Approach for Credit Scoring Problem: A Case Study of Taiwan Default Credit Card Dataset*, aparece en listas de referencias
  pero no se recuperó su contenido.
- No hice snowballing hacia adelante (citaciones) de los trabajos 1, 2 y 4. Los enlaces deben abrirse y confirmarse antes de citarlos en el informe.

## Fuentes consultadas
- [Surrogate loss for F-beta (arXiv 2104.01459)](https://arxiv.org/pdf/2104.01459)
- [Evo* 2025 Late-Breaking Abstracts (arXiv 2511.17543)](https://arxiv.org/pdf/2511.17543)
- [Ensemble Diversity Optimization for Subjective Supervision (PMLR 337)](https://proceedings.mlr.press/v337/cui26a.html) y [arXiv 2607.08493](https://arxiv.org/pdf/2607.08493)
- [Ensemble Selection from Libraries of Models (ICML 2004)](https://icml.cc/Conferences/2004/proceedings/abstracts/345.htm)
- [Munson, Cardie, Caruana (2005)](https://aclanthology.org/H05-1068/)
- [pyensemble (implementación)](https://github.com/dclambert/pyensemble)
- [sigmoidF1 (arXiv 2108.10566)](https://arxiv.org/pdf/2108.10566)
- [Lipton et al., Thresholding Classifiers to Maximize F1 (arXiv 1402.1892)](https://arxiv.org/pdf/1402.1892)
- [An Optimised Greedy-Weighted Ensemble Framework for Financial Loan Default Prediction (arXiv 2603.18927)](https://arxiv.org/abs/2603.18927)
- [TabNet-Stacking (PMC11506879)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11506879/)
- [STRIKE (arXiv 2604.17622)](https://arxiv.org/pdf/2604.17622)
- [Comparison of Different Ensemble Methods in Credit Card Default Prediction (UHD 2021)](https://journals.uhd.edu.iq/index.php/uhdjst/article/download/806/832?inline=1)
