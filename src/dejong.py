"""Validación del algoritmo genético con las funciones de prueba de De Jong (1975).

El algoritmo genético del proyecto (``src/optimization.py::run_genetic_search``) elige hiperparámetros, así
que no se puede saber si está bien configurado mirando solo el F1: el óptimo real es desconocido. Las cinco
funciones de De Jong tienen un mínimo conocido y sirven para comprobarlo.

Aquí se aplica el MISMO ciclo genético a esas funciones:
- mismos operadores (``_init_gene``, ``_cx_uniform``, ``_mutate_individual``, torneo y elitismo);
- misma configuración del proyecto (``config.GA_*``): 16 individuos, 10 generaciones, cruce 0.6, mutación
  0.3 por individuo y 0.2 por gen, torneo de 3 y élite de 2.

Lo único que cambia es la función de aptitud, que pasa a ser −f(x) porque el genético maximiza. El ciclo
replica el de ``run_genetic_search`` para no modificar el código que produjo los resultados.

Funciones de De Jong (todas se minimizan):
- F1, esfera: sum x_i², n = 3, x_i en [−5.12, 5.12], mínimo 0.
- F2, Rosenbrock: 100(x1² − x2)² + (1 − x1)², n = 2, x_i en [−2.048, 2.048], mínimo 0.
- F3, escalón: sum floor(x_i), n = 5, x_i en [−5.12, 5.12], mínimo −30.
- F4, cuártica con ruido: sum i·x_i⁴ + N(0, 1), n = 30, x_i en [−1.28, 1.28], mínimo 0 (sin ruido).
- F5, zorreras de Shekel: n = 2, x_i en [−65.536, 65.536], mínimo ≈ 0.998.

Comparación: con cada semilla, la búsqueda aleatoria recibe exactamente el mismo número de evaluaciones que
usó el genético. Se reporta el mejor valor encontrado (sin ruido en F4) y la brecha al óptimo.

Uso (desde la raíz): python -m src.dejong
"""
from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd
from deap import base, creator, tools

from src import config
from src.models import HyperparamSpec
from src.optimization import _cx_uniform, _init_gene, _mutate_individual

_A = np.array([[-32, -16, 0, 16, 32] * 5, [v for v in (-32, -16, 0, 16, 32) for _ in range(5)]], dtype=float)


def f1_esfera(x):
    return float(np.sum(np.square(x)))


def f2_rosenbrock(x):
    return float(100 * (x[0] ** 2 - x[1]) ** 2 + (1 - x[0]) ** 2)


def f3_escalon(x):
    return float(np.sum(np.floor(x)))


def f4_cuartica(x, rng=None):
    base_ = float(np.sum(np.arange(1, len(x) + 1) * np.power(x, 4)))
    return base_ + (rng.normal() if rng is not None else 0.0)


def f5_shekel(x):
    s = sum(1.0 / (j + 1 + (x[0] - _A[0, j]) ** 6 + (x[1] - _A[1, j]) ** 6) for j in range(25))
    return float(1.0 / (0.002 + s))


FUNCIONES = {  # nombre: (función, dimensión, límite, mínimo conocido)
    "F1 esfera": (f1_esfera, 3, 5.12, 0.0),
    "F2 Rosenbrock": (f2_rosenbrock, 2, 2.048, 0.0),
    "F3 escalón": (f3_escalon, 5, 5.12, -30.0),
    "F4 cuártica con ruido": (f4_cuartica, 30, 1.28, 0.0),
    "F5 Shekel": (f5_shekel, 2, 65.536, 0.998004),
}


def _espacio(n, lim):
    return [HyperparamSpec(f"x{i}", "float", low=-lim, high=lim) for i in range(n)]


def genetico(nombre, semilla, population_size=config.GA_POPULATION_SIZE, n_generations=config.GA_N_GENERATIONS):
    """Mismo ciclo de ``run_genetic_search``. Devuelve (historial del mejor valor sin ruido, n de evaluaciones)."""
    f, n, lim, _ = FUNCIONES[nombre]
    random.seed(semilla)
    ruido = np.random.default_rng(semilla)
    hps = _espacio(n, lim)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)
    tb = base.Toolbox()
    tb.register("individual", lambda: creator.Individual([_init_gene(hp) for hp in hps]))
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("mate", _cx_uniform, indpb=0.5)
    tb.register("mutate", _mutate_individual, hyperparams=hps, indpb=config.GA_MUT_INDPB)
    tb.register("select", tools.selTournament, tournsize=config.GA_TOURNAMENT_SIZE)

    limpio = []   # valor sin ruido de cada evaluación, en orden

    def evaluar(ind):
        x = np.array(ind, dtype=float)
        limpio.append(f(x) if nombre != "F4 cuártica con ruido" else f4_cuartica(x))
        y = f4_cuartica(x, ruido) if nombre == "F4 cuártica con ruido" else limpio[-1]
        return (-y,)

    pop = tb.population(n=population_size)
    for ind in pop:
        ind.fitness.values = evaluar(ind)
    for _ in range(n_generations):
        elite = [tb.clone(i) for i in tools.selBest(pop, config.GA_ELITE_SIZE)]
        hijos = [tb.clone(i) for i in tb.select(pop, len(pop) - config.GA_ELITE_SIZE)]
        for a, b in zip(hijos[::2], hijos[1::2]):
            if random.random() < config.GA_CX_PROB:
                tb.mate(a, b)
                del a.fitness.values, b.fitness.values
        for m in hijos:
            if random.random() < config.GA_MUT_PROB:
                tb.mutate(m)
                del m.fitness.values
        for ind in hijos:
            if not ind.fitness.valid:
                ind.fitness.values = evaluar(ind)
        pop = elite + hijos
    return np.minimum.accumulate(limpio), len(limpio)


def aleatoria(nombre, semilla, n_eval):
    """Búsqueda aleatoria uniforme en el mismo espacio con el mismo número de evaluaciones."""
    f, n, lim, _ = FUNCIONES[nombre]
    rng = np.random.default_rng(10_000 + semilla)
    vals = []
    for _ in range(n_eval):
        x = rng.uniform(-lim, lim, n)
        vals.append(f(x) if nombre != "F4 cuártica con ruido" else f4_cuartica(x))
    return np.minimum.accumulate(vals)


def experimento(semillas=range(10), population_size=config.GA_POPULATION_SIZE, n_generations=config.GA_N_GENERATIONS):
    filas, curvas = [], {}
    for nombre, (_, _, _, opt) in FUNCIONES.items():
        for s in semillas:
            g, n_eval = genetico(nombre, s, population_size, n_generations)
            r = aleatoria(nombre, s, n_eval)
            filas.append(dict(funcion=nombre, semilla=s, evaluaciones=n_eval, optimo=opt,
                              genetico=g[-1], aleatoria=r[-1], brecha_genetico=g[-1] - opt, brecha_aleatoria=r[-1] - opt))
            curvas.setdefault(nombre, []).append((g, r))
    return pd.DataFrame(filas), curvas


def resumen(T):
    R = T.groupby("funcion").agg(evaluaciones=("evaluaciones", "median"), optimo=("optimo", "first"),
                                 genetico_mediana=("genetico", "median"), aleatoria_mediana=("aleatoria", "median"),
                                 genetico_gana=("genetico", lambda s: 0))
    R["genetico_gana"] = T.assign(g=T.genetico < T.aleatoria).groupby("funcion").g.sum()
    return R.reindex(list(FUNCIONES))


def _figura(curvas, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.6))
    for ax, (nombre, cs) in zip(axs, curvas.items()):
        opt = FUNCIONES[nombre][3]
        m = min(len(g) for g, _ in cs)
        G = np.array([g[:m] for g, _ in cs]) - opt
        R = np.array([r[:m] for _, r in cs]) - opt
        x = np.arange(1, m + 1)
        for M, col, lab in ((G, "#B88F3D", "Genético"), (R, "#36456D", "Aleatoria")):
            ax.plot(x, np.median(M, 0), color=col, lw=2, label=lab)
            ax.fill_between(x, np.percentile(M, 25, 0), np.percentile(M, 75, 0), color=col, alpha=.18)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(nombre, fontsize=10); ax.set_xlabel("Evaluaciones"); ax.grid(alpha=.3)
    axs[0].set_ylabel("Brecha al óptimo (mejor hasta ahora)"); axs[0].legend(fontsize=8)
    fig.suptitle("Genético del proyecto (16 × 10) frente a búsqueda aleatoria con el mismo presupuesto: mediana y "
                 "rango intercuartílico en 10 semillas", fontsize=11)
    fig.tight_layout(); fig.savefig(ruta, dpi=150); plt.close(fig)


if __name__ == "__main__":
    out = config.RESULTS_DIR / "dejong"
    out.mkdir(parents=True, exist_ok=True)
    T, curvas = experimento()
    T.to_csv(out / "dejong_16x10.csv", index=False)
    R = resumen(T); R.to_csv(out / "dejong_16x10_resumen.csv"); print("Presupuesto del proyecto (16 × 10)\n", R.round(4).to_string())
    _figura(curvas, config.RESULTS_DIR / "figures" / "dejong_convergencia.png")
    T2, _ = experimento(population_size=50, n_generations=50)
    T2.to_csv(out / "dejong_50x50.csv", index=False)
    R2 = resumen(T2); R2.to_csv(out / "dejong_50x50_resumen.csv"); print("Presupuesto amplio (50 × 50)\n", R2.round(4).to_string())
