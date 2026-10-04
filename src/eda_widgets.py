"""Widgets de apoyo para ``notebooks/00_eda.ipynb``.

``show_with_interpretation`` muestra una figura de Plotly con un desplegable
de variables (``updatemenus``) y, debajo, un panel con la interpretación de
la variable elegida. El panel se actualiza solo cuando cambia el desplegable
y trae dos botones: **Interpretación básica** (palabras sencillas, sin
términos estadísticos) e **Interpretación técnica**.

El enlace entre el gráfico y el panel se hace con JavaScript que Plotly
ejecuta justo después de dibujar la figura (``post_script``), así que
funciona igual en JupyterLab y en el HTML estático de Jupyter Book, sin
necesidad de un kernel vivo.
"""

from __future__ import annotations

import html
import json
import re

NAVY = "#1B3071"
DEEP_NAVY = "#0C1A41"
WARM_GOLD = "#C9930C"
CREAM = "#F9F0DE"

MODES = (("basica", "Interpretación básica"), ("tecnica", "Interpretación técnica"))


def _inline_markdown(text: str) -> str:
    """Convierte ``**negrita**`` y ```código``` a HTML (el resto se escapa)."""
    out = html.escape(text, quote=False)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"`(.+?)`", r"<code>\1</code>", out)
    return out


_JS_TEMPLATE = """
(function () {
  var gd = document.getElementById('{plot_id}');
  if (!gd || gd.__edaInterp) { return; }
  gd.__edaInterp = true;

  var LABELS = __LABELS__;
  var TEXTS = __TEXTS__;
  var MODES = __MODES__;
  var state = { idx: 0, mode: __DEFAULT_MODE__ };

  if (!document.getElementById('eda-interp-style')) {
    var st = document.createElement('style');
    st.id = 'eda-interp-style';
    st.textContent =
      '.eda-interp{font-family:Arial,sans-serif;color:__DEEP_NAVY__;background:__CREAM__;' +
      'border-left:4px solid __GOLD__;border-radius:4px;padding:12px 16px;margin:8px 0 18px 0;box-sizing:border-box;}' +
      '.eda-interp-bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px;}' +
      '.eda-interp-title{font-weight:bold;margin-right:auto;}' +
      '.eda-interp button{font-family:inherit;font-size:13px;cursor:pointer;border-radius:16px;' +
      'padding:4px 14px;border:1px solid __NAVY__;background:#fff;color:__NAVY__;}' +
      '.eda-interp button[aria-pressed=true]{background:__NAVY__;color:#fff;}' +
      '.eda-interp-body{font-size:14px;line-height:1.55;}' +
      '.eda-interp-body code{background:rgba(12,26,65,.08);padding:1px 4px;border-radius:3px;}';
    document.head.appendChild(st);
  }

  var box = document.createElement('div');
  box.className = 'eda-interp';
  // El gráfico mide __WIDTH__px; si la pantalla es más angosta, el panel se ajusta al ancho visible
  // y queda fijo a la izquierda aunque el contenedor del gráfico tenga scroll horizontal.
  box.style.position = 'sticky';
  box.style.left = '0';
  function fit() {
    var p = box.parentNode, w = document.documentElement.clientWidth;
    while (p && p !== document.body) {
      var ox = getComputedStyle(p).overflowX;
      if (ox === 'auto' || ox === 'scroll' || ox === 'hidden') { w = p.clientWidth; break; }
      p = p.parentNode;
    }
    box.style.width = Math.max(260, Math.min(__WIDTH__, w - 4)) + 'px';
  }
  var bar = document.createElement('div');
  bar.className = 'eda-interp-bar';
  var title = document.createElement('span');
  title.className = 'eda-interp-title';
  bar.appendChild(title);
  var buttons = {};
  MODES.forEach(function (m) {
    var b = document.createElement('button');
    b.type = 'button';
    b.textContent = m[1];
    b.addEventListener('click', function () { state.mode = m[0]; render(); });
    buttons[m[0]] = b;
    bar.appendChild(b);
  });
  var body = document.createElement('div');
  body.className = 'eda-interp-body';
  box.appendChild(bar);
  box.appendChild(body);
  gd.parentNode.insertBefore(box, gd.nextSibling);
  fit();
  window.addEventListener('resize', fit);

  function render() {
    var label = LABELS[state.idx];
    title.textContent = 'Interpretación de ' + label;
    body.innerHTML = TEXTS[label][state.mode];
    MODES.forEach(function (m) {
      buttons[m[0]].setAttribute('aria-pressed', m[0] === state.mode ? 'true' : 'false');
    });
  }

  gd.on('plotly_buttonclicked', function (ev) {
    var idx = ev && typeof ev.active === 'number' ? ev.active : -1;
    if (idx < 0 && ev && ev.button) { idx = LABELS.indexOf(ev.button.label); }
    if (idx >= 0 && idx < LABELS.length) { state.idx = idx; render(); }
  });
  render();
})();
"""


def build_post_script(
    labels: list[str],
    interpretations: dict[str, dict[str, str]],
    width: int = 850,
    default_mode: str = "basica",
) -> str:
    """Genera el JavaScript que acompaña a la figura (ver ``show_with_interpretation``)."""
    missing = [lab for lab in labels if lab not in interpretations]
    if missing:
        raise KeyError(f"Faltan interpretaciones para: {missing}")
    texts = {
        lab: {mode: _inline_markdown(interpretations[lab][mode]) for mode, _ in MODES}
        for lab in labels
    }
    replacements = {
        "__LABELS__": json.dumps(labels, ensure_ascii=False),
        "__TEXTS__": json.dumps(texts, ensure_ascii=False),
        "__MODES__": json.dumps([list(m) for m in MODES], ensure_ascii=False),
        "__DEFAULT_MODE__": json.dumps(default_mode),
        "__WIDTH__": str(int(width)),
        "__DEEP_NAVY__": DEEP_NAVY,
        "__NAVY__": NAVY,
        "__GOLD__": WARM_GOLD,
        "__CREAM__": CREAM,
    }
    script = _JS_TEMPLATE
    for key, value in replacements.items():
        script = script.replace(key, value)
    return script


def show_with_interpretation(
    fig,
    labels: list[str],
    interpretations: dict[str, dict[str, str]],
    default_mode: str = "basica",
) -> None:
    """Muestra ``fig`` con un panel de interpretación enlazado a su desplegable.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figura con un ``updatemenus`` tipo desplegable cuyos botones están en
        el mismo orden que ``labels``.
    labels : list[str]
        Nombre de cada opción del desplegable (p. ej. las columnas).
    interpretations : dict
        ``{label: {"basica": texto, "tecnica": texto}}``. Admite
        ``**negrita**`` y ```código```.
    default_mode : {"basica", "tecnica"}
        Interpretación que se muestra al abrir.
    """
    width = fig.layout.width or 850
    script = build_post_script(labels, interpretations, width=width, default_mode=default_mode)
    fig.show(post_script=[script])
