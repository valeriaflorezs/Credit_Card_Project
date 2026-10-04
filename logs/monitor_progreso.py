"""Genera logs/progreso.html: tablero de avance de la corrida de experimentos.

Lee results/experiments_master.csv (lo que ya terminó) y deduce qué combinación
va corriendo (la primera pendiente en el orden del orquestador). Se vuelve a
generar cada ``INTERVALO`` segundos y la página se refresca sola.

Uso (desde la raíz del proyecto, con el entorno ``creditcard``):
    python logs/monitor_progreso.py          # bucle, Ctrl+C para salir
    python logs/monitor_progreso.py --una-vez
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# Valores copiados de src/balancing.py, src/optimization.py y src/experiment_runner.py. No se importan
# esos módulos a propósito: cargarían sklearn/XGBoost (~250 MB) en un proceso que solo dibuja una página.
BALANCING_TECHNIQUES = ("none", "smote", "adasyn", "class_weight")
OPTIMIZATION_METHODS = ("grid_search", "random_search", "bayesian_optuna", "genetic_deap")
MODEL_COST_ORDER = ("naive_bayes", "logistic_regression", "decision_tree", "xgboost", "random_forest", "knn", "svm")

INTERVALO = 20
SALIDA = ROOT / "logs" / "progreso.html"
MAESTRA = ROOT / "results" / "experiments_master.csv"
MAESTRA_V1 = ROOT / "results" / "experiments_master_v1_dataset_anterior.csv"
COLAB_CSV = ROOT / "results" / "svm_gpu_results_colab.csv"  # SVM corrido en Colab, aún sin pegar a la tabla maestra
LANZADOR_LOG = ROOT / "logs" / "04_v2_lanzador.log"
MODELOS_COLAB = {"svm"}  # se corren fuera de esta máquina

NOMBRES = {
    "naive_bayes": "Naive Bayes", "logistic_regression": "Regresión logística", "decision_tree": "Árbol de decisión",
    "xgboost": "XGBoost", "random_forest": "Random Forest", "knn": "KNN", "svm": "SVM",
}
TECNICAS = {"none": "Sin balanceo", "smote": "SMOTE", "adasyn": "ADASYN", "class_weight": "Pesos de clase"}
METODOS = {"grid_search": "Grid", "random_search": "Random", "bayesian_optuna": "Bayesiana", "genetic_deap": "Genética"}
# Rampa secuencial azul (pasos 100 -> 700 de la paleta de referencia) para el F1 de las celdas terminadas.
RAMPA = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def fmt_dur(seg: float) -> str:
    seg = int(max(seg, 0))
    h, r = divmod(seg, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {s:02d} s"
    return f"{s} s"


def ram_libre_gb() -> float | None:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        return int(out) / 1024 / 1024
    except Exception:
        return None


def procesos_python() -> int:
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/NH"], capture_output=True, text=True, timeout=20).stdout
        return max(len(re.findall(r"python\.exe", out)) - 1, 0)  # sin contar este mismo monitor
    except Exception:
        return 0


def inicio_corrida() -> datetime | None:
    try:
        m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) intento 1", LANZADOR_LOG.read_text(encoding="utf-8", errors="ignore"))
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else None
    except Exception:
        return None


def intento_actual() -> str:
    try:
        intentos = re.findall(r"intento (\d+):", LANZADOR_LOG.read_text(encoding="utf-8", errors="ignore"))
        return intentos[-1] if intentos else "?"
    except Exception:
        return "?"


def color_f1(f1: float, lo: float, hi: float) -> tuple[str, str]:
    t = 0.0 if hi <= lo else (f1 - lo) / (hi - lo)
    idx = min(int(t * (len(RAMPA) - 1) + 0.5), len(RAMPA) - 1)
    fondo = RAMPA[idx]
    return fondo, ("#ffffff" if idx >= 6 else "#0b0b0b")


def generar() -> None:
    ahora = datetime.now()
    hechas = pd.read_csv(MAESTRA) if MAESTRA.exists() else pd.DataFrame(columns=["model", "technique", "method", "total_time_seconds"])
    v1 = pd.read_csv(MAESTRA_V1) if MAESTRA_V1.exists() else pd.DataFrame(columns=hechas.columns)
    # Filas de Colab: van al principio (así "últimas terminadas" muestra las locales) y solo si aún no están en la maestra.
    claves_colab: set = set()
    if COLAB_CSV.exists():
        colab = pd.read_csv(COLAB_CSV)
        ya = set(zip(hechas.model, hechas.technique, hechas.method))
        nuevas = colab[[(r.model, r.technique, r.method) not in ya for r in colab.itertuples()]]
        claves_colab = set(zip(nuevas.model, nuevas.technique, nuevas.method))
        hechas = pd.concat([nuevas, hechas], ignore_index=True)
    modelos = [m for m in MODEL_COST_ORDER]
    orden = [(m, t, me) for m in modelos for t in BALANCING_TECHNIQUES for me in OPTIMIZATION_METHODS]
    locales = [c for c in orden if c[0] not in MODELOS_COLAB]
    por_clave = {(r.model, r.technique, r.method): r for r in hechas.itertuples()}
    t_v1 = {(r.model, r.technique, r.method): r.total_time_seconds for r in v1.itertuples()}

    pendientes = [c for c in locales if c not in por_clave]
    en_curso = pendientes[0] if pendientes else None
    n_hechas = sum(1 for c in locales if c in por_clave)
    n_total = len(locales)

    # tiempos
    t0 = inicio_corrida()
    transcurrido = (ahora - t0).total_seconds() if t0 else None
    mtime = datetime.fromtimestamp(MAESTRA.stat().st_mtime) if MAESTRA.exists() else None
    desde_ultima = (ahora - mtime).total_seconds() if mtime else None
    desde_inicio_actual = desde_ultima if desde_ultima is not None else transcurrido

    # ETA (gruesa): tiempo v1 de lo pendiente x razon observada (v2 / v1) en lo ya hecho
    comparables = [c for c in locales if c in por_clave and c in t_v1]
    eta = None
    if comparables and pendientes:
        recientes = [c for c in comparables if en_curso and c[0] == en_curso[0]]
        base = recientes if len(recientes) >= 3 else comparables
        razon = sum(por_clave[c].total_time_seconds for c in base) / max(sum(t_v1[c] for c in base), 1e-9)
        eta = sum(t_v1.get(c, 0) for c in pendientes) * razon
        if en_curso and desde_inicio_actual:
            eta = max(eta - min(desde_inicio_actual, t_v1.get(en_curso, 0) * razon), 0)

    n_py = procesos_python()
    if n_hechas >= n_total:
        estado, estado_cls, icono = "Corrida completa", "ok", "✓"
    elif n_py >= 2:
        estado, estado_cls, icono = "En ejecución", "run", "▶"
    else:
        estado, estado_cls, icono = "Detenida: no hay procesos de Python activos", "bad", "✕"

    n_colab = sum(1 for c in orden if c[0] in MODELOS_COLAB and c in por_clave)
    t_colab = sum(por_clave[c].total_time_seconds for c in orden if c[0] in MODELOS_COLAB and c in por_clave)
    ram = ram_libre_gb()
    f1s = [por_clave[c].f1_mean for c in locales if c in por_clave]
    lo, hi = (min(f1s), max(f1s)) if f1s else (0, 1)

    # ---------------- HTML ----------------
    def kpi(titulo: str, valor: str, nota: str = "") -> str:
        return f'<div class="kpi"><div class="kpi-t">{titulo}</div><div class="kpi-v">{valor}</div><div class="kpi-n">{nota}</div></div>'

    kpis = [
        kpi("Combinaciones listas", f"{n_hechas} / {n_total}", f"{100 * n_hechas / n_total:.0f}% de los 6 modelos locales"),
        kpi("Tiempo transcurrido", fmt_dur(transcurrido) if transcurrido is not None else "—", f"intento {intento_actual()} de 6"),
        kpi("Última fila guardada", f"hace {fmt_dur(desde_ultima)}" if desde_ultima is not None else "—", "una combinación larga puede tardar más de 30 min"),
        kpi("Tiempo restante (estimado)", f"≈ {fmt_dur(eta)}" if eta is not None else "—", "aproximado; se afina con cada modelo"),
        kpi("SVM (Colab, GPU)", f"{n_colab} / 16", f"{fmt_dur(t_colab)} de GPU en total" if n_colab else "pendiente"),
        kpi("RAM libre", f"{ram:.1f} GB" if ram is not None else "—", "menos de 1.5 GB es riesgo de caída"),
    ]

    # barras por modelo
    barras = []
    for m in modelos:
        total_m = len(BALANCING_TECHNIQUES) * len(OPTIMIZATION_METHODS)
        if m in MODELOS_COLAB and not any((m, t, me) in por_clave for t in BALANCING_TECHNIQUES for me in OPTIMIZATION_METHODS):
            barras.append(f'<div class="bar-row"><div class="bar-n">{NOMBRES[m]}</div><div class="bar colab"><span>Se corre en Colab (GPU)</span></div><div class="bar-c">0 / {total_m}</div></div>')
            continue
        n = sum(1 for c in orden if c[0] == m and c in por_clave)
        cls = "bar-fill done" if n == total_m else "bar-fill"
        actual = en_curso and en_curso[0] == m
        marca = (" ▶" if actual else "") + (" · Colab" if m in MODELOS_COLAB else "")
        horas = sum(por_clave[c].total_time_seconds for c in orden if c[0] == m and c in por_clave)
        barras.append(
            f'<div class="bar-row"><div class="bar-n">{NOMBRES[m]}{marca}</div>'
            f'<div class="bar"><div class="{cls}" style="width:{100 * n / total_m:.1f}%"></div></div>'
            f'<div class="bar-c">{n} / {total_m} · {fmt_dur(horas) if n else "—"}</div></div>'
        )

    # cuadrículas por modelo
    tablas = []
    for m in modelos:
        filas = []
        for t in BALANCING_TECHNIQUES:
            celdas = [f'<th scope="row">{TECNICAS.get(t, t)}</th>']
            for me in OPTIMIZATION_METHODS:
                c = (m, t, me)
                if m in MODELOS_COLAB and c not in por_clave:
                    celdas.append('<td class="colab" title="Se corre en Colab">Colab</td>')
                elif c in por_clave:
                    r = por_clave[c]
                    fondo, texto = color_f1(r.f1_mean, lo, hi)
                    tip = (f"{NOMBRES[m]} / {TECNICAS.get(t, t)} / {METODOS.get(me, me)}\n"
                           f"F1 {r.f1_mean:.4f} ± {r.f1_std:.4f}\nAUC {r.roc_auc_mean:.4f}\nRecall {r.recall_mean:.4f}\nPrecisión {r.precision_mean:.4f}\n"
                           f"Accuracy {r.accuracy_mean:.4f}\nTiempo {fmt_dur(r.total_time_seconds)}"
                           + ("\nCorrido en Colab (GPU)" if c in claves_colab else ""))
                    celdas.append(f'<td class="hecha" style="background:{fondo};color:{texto}" title="{html.escape(tip)}"><b>✓ {r.f1_mean:.3f}</b><small>{fmt_dur(r.total_time_seconds)}</small></td>')
                elif c == en_curso:
                    celdas.append(f'<td class="curso" title="Combinación en curso"><b>▶ en curso</b><small>{fmt_dur(desde_inicio_actual) if desde_inicio_actual is not None else ""}</small></td>')
                else:
                    celdas.append('<td class="pend">pendiente</td>')
            filas.append("<tr>" + "".join(celdas) + "</tr>")
        encabezado = "<tr><th></th>" + "".join(f"<th>{METODOS.get(me, me)}</th>" for me in OPTIMIZATION_METHODS) + "</tr>"
        n = sum(1 for c in orden if c[0] == m and c in por_clave)
        total_sub = len(BALANCING_TECHNIQUES) * len(OPTIMIZATION_METHODS)
        if m in MODELOS_COLAB:
            sub = "Colab (GPU), pendiente" if n == 0 else f"{n} / {total_sub} · Colab (GPU)"
        else:
            sub = f"{n} / {total_sub}"
        tablas.append(f'<section class="modelo"><h3>{NOMBRES[m]} <span class="sub">{sub}</span></h3><table>{encabezado}{"".join(filas)}</table></section>')

    # últimas terminadas
    ult = hechas.tail(8).iloc[::-1]
    filas_ult = "".join(
        f"<tr><td>{NOMBRES.get(r.model, r.model)}</td><td>{TECNICAS.get(r.technique, r.technique)}</td><td>{METODOS.get(r.method, r.method)}</td>"
        f"<td class='num'>{r.f1_mean:.3f}</td><td class='num'>{r.roc_auc_mean:.3f}</td><td class='num'>{fmt_dur(r.total_time_seconds)}</td></tr>"
        for r in ult.itertuples()
    )
    tabla_ult = ("<table class='ult'><tr><th>Modelo</th><th>Técnica</th><th>Método</th><th>F1</th><th>AUC</th><th>Tiempo</th></tr>" + filas_ult + "</table>") if len(ult) else "<p>Aún no hay filas.</p>"

    escala = (f'<div class="escala"><span>F1 {lo:.2f}</span><div class="grad" style="background:linear-gradient(90deg,{",".join(RAMPA)})"></div><span>{hi:.2f}</span></div>') if f1s else ""

    pagina = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="{INTERVALO}">
<title>Progreso de la corrida</title>
<style>
:root {{ color-scheme: light; --bg:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --line:#dcdbd5; --pend:#f0efec; --ok:#0ca30c; --run:#fab219; --bad:#d03b3b; --bar:#2a78d6; --bar-done:#0ca30c; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --bg:#1a1a19; --card:#232322; --ink:#ffffff; --ink2:#c3c2b7; --line:#3a3a37; --pend:#2c2c2a; --bar:#3987e5; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:20px 16px 40px; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui,"Segoe UI",Arial,sans-serif; }}
main {{ max-width:1180px; margin:0 auto; }}
h1 {{ font-size:20px; margin:0 0 4px; }} h2 {{ font-size:15px; margin:26px 0 10px; }} h3 {{ font-size:14px; margin:0 0 6px; }}
.top {{ display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center; justify-content:space-between; }}
.estado {{ display:inline-flex; gap:6px; align-items:center; padding:4px 12px; border-radius:16px; font-weight:600; border:1.5px solid; }}
.estado.run {{ border-color:var(--run); }} .estado.ok {{ border-color:var(--ok); }} .estado.bad {{ border-color:var(--bad); }}
.stamp {{ color:var(--ink2); font-size:12px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; margin-top:14px; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
.kpi-t {{ color:var(--ink2); font-size:12px; }} .kpi-v {{ font-size:22px; font-weight:700; margin:2px 0; }} .kpi-n {{ color:var(--ink2); font-size:11px; }}
.bar-row {{ display:grid; grid-template-columns:170px 1fr 190px; gap:10px; align-items:center; margin:6px 0; }}
.bar {{ background:var(--pend); border-radius:4px; height:14px; overflow:hidden; border:1px solid var(--line); }}
.bar-fill {{ background:var(--bar); height:100%; border-radius:0 4px 4px 0; }} .bar-fill.done {{ background:var(--bar-done); }}
.bar.colab {{ background:repeating-linear-gradient(135deg,var(--pend) 0 6px,transparent 6px 12px); font-size:11px; color:var(--ink2); line-height:12px; padding-left:8px; }}
.bar-c {{ color:var(--ink2); font-size:12px; text-align:right; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:14px; }}
.modelo {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
.sub {{ color:var(--ink2); font-weight:400; font-size:12px; margin-left:6px; }}
table {{ border-collapse:separate; border-spacing:2px; width:100%; table-layout:fixed; }}
th {{ font-weight:600; font-size:11px; color:var(--ink2); text-align:center; padding:2px; }} th[scope=row] {{ text-align:left; width:92px; }}
td {{ text-align:center; padding:6px 2px; border-radius:4px; vertical-align:middle; }} td b {{ display:block; font-size:13px; }} td small {{ display:block; font-size:10px; opacity:.85; }}
td.pend {{ background:var(--pend); color:var(--ink2); font-size:11px; }} td.colab {{ background:repeating-linear-gradient(135deg,var(--pend) 0 6px,transparent 6px 12px); color:var(--ink2); font-size:11px; }}
td.curso {{ background:var(--run); color:#0b0b0b; animation:latido 1.6s ease-in-out infinite; }}
@keyframes latido {{ 50% {{ opacity:.65; }} }} @media (prefers-reduced-motion: reduce) {{ td.curso {{ animation:none; }} }}
table.ult th {{ text-align:left; }} table.ult td {{ text-align:left; background:none; padding:3px 4px; border-bottom:1px solid var(--line); border-radius:0; }} table.ult td.num {{ text-align:right; }}
.escala {{ display:flex; align-items:center; gap:8px; font-size:11px; color:var(--ink2); margin:8px 0 0; max-width:360px; }} .grad {{ flex:1; height:8px; border-radius:4px; }}
.nota {{ color:var(--ink2); font-size:12px; margin-top:18px; }}
@media (max-width:640px) {{ .bar-row {{ grid-template-columns:110px 1fr; }} .bar-c {{ grid-column:1 / -1; text-align:left; }} .grid {{ grid-template-columns:1fr; }} }}
</style></head>
<body><main>
<div class="top"><div><h1>Progreso de la corrida de experimentos</h1><div class="stamp">Actualizado {ahora:%H:%M:%S} · se refresca solo cada {INTERVALO} s · datos: results/experiments_master.csv</div></div>
<div class="estado {estado_cls}"><span aria-hidden="true">{icono}</span> {estado}</div></div>
<div class="kpis">{"".join(kpis)}</div>
<h2>Avance por modelo</h2>{"".join(barras)}
<h2>Cada combinación <small style="font-weight:400;color:var(--ink2)">(técnica de balanceo × método de optimización; color = F1, pasa el cursor para ver todas las métricas)</small></h2>
{escala}
<div class="grid" style="margin-top:10px">{"".join(tablas)}</div>
<h2>Últimas terminadas</h2>{tabla_ult}
<p class="nota">La combinación «en curso» se deduce como la primera pendiente en el orden de ejecución. El tiempo restante es una estimación basada en la corrida anterior escalada por la velocidad observada; los modelos de árboles y XGBoost suelen aprovechar mejor el paralelismo, así que puede cambiar mucho al llegar a ellos. SVM se corrió aparte en Colab con GPU (sus tiempos no son comparables con los locales); la estimación de tiempo restante solo cuenta los modelos locales.</p>
</main></body></html>"""
    tmp = SALIDA.with_suffix(".tmp")
    tmp.write_text(pagina, encoding="utf-8")
    tmp.replace(SALIDA)


if __name__ == "__main__":
    una_vez = "--una-vez" in sys.argv
    while True:
        try:
            generar()
        except Exception as exc:  # no tumbar el monitor por un error de lectura puntual (archivo en escritura)
            print(f"{datetime.now():%H:%M:%S} aviso: {exc!r}", flush=True)
        if una_vez:
            break
        time.sleep(INTERVALO)
