"""Reporte HTML del progreso de una corrida de ``run_stacking_repeated.py``.

Lee ``<salida>/metricas.csv`` y ``<salida>/configuracion.json`` (y, si se da, el log de la corrida) y
genera una página autocontenida con:

- pliegues terminados frente al total, barra de progreso y tiempo restante estimado;
- estado de la corrida (corriendo, detenida o terminada) y la última actividad del log;
- F1 medio **provisional** de cada modelo hasta ahora;
- las últimas líneas del log.

A propósito **no** muestra p-valores: mirar la significancia a mitad de camino y detener la corrida
según ella invalidaría el pre-registro.

Uso::

    python -m src.progreso --salida results/redes_repeated --log redes_log.txt          # una vez
    python -m src.progreso --salida results/stacking_repeated --log log.txt --cada 60   # cada 60 s

En Colab: ``display(HTML(generar_html(...)))`` (ver ``notebooks/14_redes_neuronales_colab.ipynb``).
"""
from __future__ import annotations

import argparse
import html
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ESTILO = """
<style>
.prog{--fondo:#ffffff;--texto:#1b1f24;--suave:#5b6470;--borde:#d9dee4;--pista:#eef1f4;--barra:#3b6ea8;
--propio:#b5832f;--ok:#2e7d4f;--alerta:#b54708;--fondo2:#f7f8fa;
font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:var(--texto);background:var(--fondo);
max-width:880px;margin:0 auto;padding:20px 16px;line-height:1.45;box-sizing:border-box}
@media (prefers-color-scheme:dark){.prog{--fondo:#16191d;--texto:#e6e9ed;--suave:#9aa4b0;--borde:#2e343b;
--pista:#262b31;--barra:#6b9bd1;--propio:#d9a650;--ok:#5cbf86;--alerta:#f0a35c;--fondo2:#1d2126}}
.prog *{box-sizing:border-box}
.prog h1{font-size:1.25rem;margin:0 0 2px}
.prog .sub{color:var(--suave);font-size:.85rem;margin:0 0 16px}
.prog .tarjetas{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}
.prog .tarjeta{border:1px solid var(--borde);border-radius:8px;padding:10px 12px;background:var(--fondo2)}
.prog .tarjeta b{display:block;font-size:1.3rem;font-variant-numeric:tabular-nums}
.prog .tarjeta span{color:var(--suave);font-size:.8rem}
.prog .pista{height:12px;background:var(--pista);border-radius:6px;overflow:hidden;margin:4px 0 18px}
.prog .relleno{height:100%;background:var(--barra)}
.prog .estado{display:inline-block;padding:1px 8px;border-radius:10px;font-size:.8rem;font-weight:600;
border:1px solid currentColor}
.prog .corriendo{color:var(--ok)}.prog .detenida{color:var(--alerta)}.prog .terminada{color:var(--barra)}
.prog table{width:100%;border-collapse:collapse;font-size:.88rem;font-variant-numeric:tabular-nums}
.prog th,.prog td{text-align:left;padding:6px 6px;border-bottom:1px solid var(--borde);vertical-align:middle}
.prog th{color:var(--suave);font-weight:600;font-size:.78rem}
.prog .num{text-align:right;white-space:nowrap}
.prog .barra-f1{height:8px;background:var(--barra);border-radius:4px;min-width:2px}
.prog tr.propio td:first-child{font-weight:700;color:var(--propio)}
.prog tr.propio .barra-f1{background:var(--propio)}
.prog .aviso{font-size:.8rem;color:var(--suave);border-left:3px solid var(--alerta);padding:6px 10px;
margin:12px 0;background:var(--fondo2)}
.prog pre{background:var(--fondo2);border:1px solid var(--borde);border-radius:8px;padding:10px;font-size:.75rem;
overflow-x:auto;white-space:pre-wrap;word-break:break-word;margin:0}
.prog h2{font-size:1rem;margin:20px 0 8px}
.prog .tabla{overflow-x:auto}
</style>
"""
PROPIOS = {"tabm_fisher", "cs_fof", "stack_fwls_con"}  # modelos que el equipo propone (se resaltan)


def _duracion(seg: float) -> str:
    seg = max(0, int(seg))
    h, m = divmod(seg // 60, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def generar_html(salida: Path, log: Path | None = None, corriendo: bool | None = None, refrescar: int = 0) -> str:
    salida = Path(salida)
    conf = {}
    if (salida / "configuracion.json").exists():
        conf = json.loads((salida / "configuracion.json").read_text(encoding="utf-8"))
    total = int(conf.get("repeticiones", 10)) * int(conf.get("pliegues", 5))
    tabla = pd.read_csv(salida / "metricas.csv") if (salida / "metricas.csv").exists() else pd.DataFrame()
    hechos = tabla.groupby(["repeticion", "pliegue"]).ngroups if len(tabla) else 0

    # Tiempo por pliegue: suma del tiempo de los modelos base de cada (repetición, pliegue).
    seg_pliegue = 0.0
    if len(tabla):
        base = tabla[tabla.tipo == "base"] if "tipo" in tabla else tabla
        seg_pliegue = float(base.groupby(["repeticion", "pliegue"]).segundos.sum().mean())
    restante = seg_pliegue * (total - hechos) if hechos else None

    # Estado a partir del log.
    lineas, ultima = [], None
    if log and Path(log).exists():
        lineas = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()
        ultima = datetime.fromtimestamp(Path(log).stat().st_mtime)
    if hechos >= total:
        estado = ("terminada", "Terminada")
    elif corriendo is None:
        reciente = ultima is not None and (datetime.now() - ultima).total_seconds() < max(900, 2 * seg_pliegue)
        estado = ("corriendo", "Corriendo") if reciente else ("detenida", "Sin actividad reciente")
    else:
        estado = ("corriendo", "Corriendo") if corriendo else ("detenida", "Detenida: relanzar para continuar")
    errores = [l for l in lineas if "Traceback" in l or "Error" in l]

    filas = ""
    if len(tabla):
        res = (tabla.groupby("candidato")
               .agg(n=("f1_umbral", "size"), f1=("f1_umbral", "mean"), sd=("f1_umbral", "std"),
                    auc=("roc_auc", "mean"), seg=("segundos", "mean"))
               .sort_values("f1", ascending=False))
        lo, hi = res.f1.min() - 0.01, res.f1.max()  # margen para que el peor no quede en cero
        for nombre, r in res.iterrows():
            ancho = 8 + 92 * ((r.f1 - lo) / (hi - lo) if hi > lo else 1)
            clase = ' class="propio"' if nombre in PROPIOS else ""
            sd = "–" if pd.isna(r.sd) else f"{r.sd:.4f}"
            filas += (f"<tr{clase}><td>{html.escape(str(nombre))}</td><td class='num'>{int(r.n)}</td>"
                      f"<td class='num'>{r.f1:.4f}</td><td class='num'>{sd}</td>"
                      f"<td style='width:30%'><div class='barra-f1' style='width:{ancho:.0f}%'></div></td>"
                      f"<td class='num'>{r.auc:.4f}</td><td class='num'>{r.seg:.0f} s</td></tr>")
    tabla_html = (f"<div class='tabla'><table><thead><tr><th>Modelo</th><th class='num'>Pliegues</th>"
                  f"<th class='num'>F1 medio</th><th class='num'>DE</th><th>F1 relativo</th>"
                  f"<th class='num'>AUC-ROC</th><th class='num'>Tiempo/pliegue</th></tr></thead>"
                  f"<tbody>{filas}</tbody></table></div>") if filas else "<p>Todavía no termina ningún pliegue.</p>"

    pct = 100 * hechos / total if total else 0
    cola = "\n".join(lineas[-12:]) if lineas else "(sin log)"
    meta = f'<meta http-equiv="refresh" content="{refrescar}">' if refrescar else ""
    nombre = salida.name
    error_html = (f"<div class='aviso'><b>Hay errores en el log</b> ({len(errores)} líneas). "
                  f"Última: {html.escape(errores[-1][:200])}</div>") if errores else ""
    return f"""{meta}{ESTILO}
<div class="prog">
  <h1>Progreso de la corrida: {html.escape(nombre)}</h1>
  <p class="sub">Actualizado {datetime.now():%Y-%m-%d %H:%M:%S} · <span class="estado {estado[0]}">{estado[1]}</span>
  {f"· última actividad del log {ultima:%H:%M:%S}" if ultima else ""}</p>
  <div class="tarjetas">
    <div class="tarjeta"><b>{hechos} / {total}</b><span>pliegues terminados</span></div>
    <div class="tarjeta"><b>{pct:.0f} %</b><span>avance</span></div>
    <div class="tarjeta"><b>{_duracion(seg_pliegue) if hechos else "–"}</b><span>por pliegue (promedio)</span></div>
    <div class="tarjeta"><b>{_duracion(restante) if restante is not None else "–"}</b><span>tiempo restante estimado</span></div>
  </div>
  <div class="pista"><div class="relleno" style="width:{pct:.1f}%"></div></div>
  {error_html}
  <h2>F1 provisional por modelo</h2>
  {tabla_html}
  <div class="aviso">Cifras provisionales, sin pruebas de significancia. No detengan ni cambien la corrida según
  ellas: el protocolo pre-registrado exige terminar las {total} particiones y luego correr el análisis.</div>
  <h2>Últimas líneas del log</h2>
  <pre>{html.escape(cola)}</pre>
</div>"""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=Path("results/redes_repeated"))
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--html", type=Path, default=None, help="por defecto <salida>/progreso.html")
    ap.add_argument("--cada", type=int, default=0, help="segundos entre actualizaciones (0 = una sola vez)")
    args = ap.parse_args(argv)
    destino = args.html or args.salida / "progreso.html"
    while True:
        cuerpo = generar_html(args.salida, args.log, refrescar=args.cada or 0)
        destino.write_text(f"<!doctype html><html lang='es'><head><meta charset='utf-8'>"
                           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
                           f"<title>Progreso de la corrida</title><style>body{{margin:0;background:#fff}}"
                           f"@media (prefers-color-scheme:dark){{body{{background:#16191d}}}}</style></head>"
                           f"<body>{cuerpo}</body></html>",
                           encoding="utf-8")
        print(f"{datetime.now():%H:%M:%S} reporte actualizado: {destino}", flush=True)
        if not args.cada:
            break
        time.sleep(args.cada)


if __name__ == "__main__":
    main()
