"""Componentes visuales reutilizables del dashboard Netflix · Churn Analytics.

Todas las páginas usan estas funciones en lugar de escribir HTML o CSS sueltos, así
el diseño es igual en todo el dashboard y se cambia en un solo lugar.

Uso:
    import ui
    ui.aplicar_estilos()
    ui.encabezado_pagina("Panorama del abandono", "Descripción...", etiqueta="Resumen ejecutivo")
    ui.fila_kpis([("Clientes analizados", "5,000", "base completa"), ...])
    ui.seccion("¿Qué modelo se seleccionó?")
    ui.nota("Correlación no significa causalidad.")
    ui.pie(["notebook netflix_churned.ipynb", "dataset del proyecto"])
"""
from html import escape

import streamlit as st

# ─── Paleta (tokens de diseño) ──────────────────────────────────────────────
# Un solo lugar para los colores: si cambia la identidad visual, se cambia aquí.
COLORES = {
    "rojo": "#E50914",       # acento: lo que requiere atención
    "fondo": "#111318",      # fondo de la página (igual que .streamlit/config.toml)
    "tarjeta": "#1B1F27",    # fondo de tarjetas
    "borde": "#313640",
    "texto": "#F5F6FA",
    "texto_suave": "#B9BEC8",
    "verde": "#46D369",      # cumplimiento de objetivos
    "ambar": "#F5A524",      # advertencia
}

_TONOS = {"normal": COLORES["texto"], "alerta": COLORES["rojo"], "ok": COLORES["verde"], "aviso": COLORES["ambar"]}

_CSS = f"""
<style>
:root {{
  --rojo: {COLORES['rojo']}; --tarjeta: {COLORES['tarjeta']}; --borde: {COLORES['borde']};
  --texto: {COLORES['texto']}; --texto-suave: {COLORES['texto_suave']};
  --verde: {COLORES['verde']}; --ambar: {COLORES['ambar']};
}}
.block-container {{padding-top: 5rem; max-width: 1300px;}}
h1, h2, h3 {{letter-spacing: -.02em;}}

/* Marca */
.ui-marca {{font-weight: 900; font-size: 1.35rem; color: var(--rojo); letter-spacing: .12em; line-height: 1;}}
.ui-marca span {{color: var(--texto); font-size: .95rem; font-weight: 500; letter-spacing: 0;}}
.ui-marca-sub {{color: var(--texto-suave); font-size: .72rem; letter-spacing: .08em; margin-top: .25rem;}}

/* Encabezado de página */
.ui-encabezado {{margin: .4rem 0 1.1rem 0; padding-bottom: .9rem; border-bottom: 1px solid var(--borde);}}
.ui-etiqueta {{color: var(--rojo); font-size: .74rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;}}
.ui-encabezado h1 {{font-size: 2.1rem; margin: .15rem 0 .35rem 0; padding: 0;}}
.ui-encabezado p {{color: var(--texto-suave); font-size: 1rem; margin: 0; max-width: 70ch;}}

/* Título de sección con barra de acento */
.ui-seccion {{display: flex; align-items: center; gap: .6rem; margin: 1.6rem 0 .3rem 0;}}
.ui-seccion::before {{content: ""; width: 4px; height: 1.3rem; background: var(--rojo); border-radius: 2px;}}
.ui-seccion h3 {{margin: 0; padding: 0; font-size: 1.3rem;}}
.ui-seccion-ayuda {{color: var(--texto-suave); font-size: .88rem; margin: 0 0 .8rem 0;}}

/* Tarjetas KPI */
.ui-kpi {{background: var(--tarjeta); border: 1px solid var(--borde); border-radius: 12px;
  padding: 14px 18px; height: 100%; min-height: 108px; display: flex; flex-direction: column; gap: .2rem;}}
.ui-kpi-etiqueta {{color: var(--texto-suave); font-size: .82rem;}}
.ui-kpi-valor {{font-size: 1.85rem; font-weight: 700; line-height: 1.15; font-variant-numeric: tabular-nums;}}
.ui-kpi-nota {{color: var(--texto-suave); font-size: .75rem; margin-top: auto;}}

/* Tarjetas de texto (hallazgos) */
.ui-tarjeta {{background: var(--tarjeta); border: 1px solid var(--borde); border-radius: 12px;
  padding: 14px 18px; height: 100%;}}
.ui-tarjeta h4 {{margin: 0 0 .5rem 0; font-size: 1rem;}}
.ui-tarjeta li, .ui-tarjeta p {{font-size: .9rem; color: #D5D9E0; margin-bottom: .3rem;}}
.ui-tarjeta ul {{padding-left: 1.1rem; margin: 0;}}

/* Notas de interpretación */
.ui-nota {{border-left: 3px solid var(--texto-suave); background: rgba(255,255,255,.04);
  padding: 10px 14px; border-radius: 0 8px 8px 0; color: #D5D9E0; font-size: .9rem; margin: .6rem 0;}}
.ui-nota.aviso {{border-left-color: var(--ambar);}}
.ui-nota.ok {{border-left-color: var(--verde);}}
.ui-nota.alerta {{border-left-color: var(--rojo);}}

/* Pie de página */
.ui-pie {{border-top: 1px solid var(--borde); margin-top: 2.2rem; padding-top: .8rem; color: var(--texto-suave);
  font-size: .78rem; display: flex; flex-wrap: wrap; justify-content: space-between; gap: .4rem 1.5rem;}}

/* st.metric que todavía se usa en algunas páginas: mismo aspecto que las tarjetas */
div[data-testid="stMetric"] {{background: var(--tarjeta); padding: 14px 18px; border: 1px solid var(--borde); border-radius: 12px;}}
div[data-testid="stMetricValue"] {{font-size: 1.85rem;}}
div[data-testid="stMetric"] * {{color: var(--texto) !important;}}
</style>
"""


def _html(contenido: str, contenedor=None) -> None:
    (contenedor or st).markdown(contenido, unsafe_allow_html=True)


def aplicar_estilos() -> None:
    """Carga el CSS del dashboard. Se llama una vez, después de st.set_page_config."""
    _html(_CSS)


def marca(subtitulo: str = "CHURN ANALYTICS") -> None:
    """Marca en la barra lateral."""
    _html(f'<div class="ui-marca">NETFLIX</div><div class="ui-marca-sub">{escape(subtitulo)}</div>', st.sidebar)


def encabezado_pagina(titulo: str, descripcion: str = "", etiqueta: str = "") -> None:
    """Encabezado común: etiqueta pequeña en rojo, título y descripción de una línea."""
    eti = f'<div class="ui-etiqueta">{escape(etiqueta)}</div>' if etiqueta else ""
    desc = f"<p>{escape(descripcion)}</p>" if descripcion else ""
    _html(f'<div class="ui-encabezado">{eti}<h1>{escape(titulo)}</h1>{desc}</div>')


def seccion(titulo: str, ayuda: str = "") -> None:
    """Título de sección con barra de acento; reemplaza a st.subheader."""
    _html(f'<div class="ui-seccion"><h3>{escape(titulo)}</h3></div>')
    if ayuda:
        _html(f'<p class="ui-seccion-ayuda">{escape(ayuda)}</p>')


def tarjeta_kpi(etiqueta: str, valor: str, nota: str = "", tono: str = "normal", contenedor=None) -> None:
    """Tarjeta de indicador. tono: 'normal', 'alerta' (rojo), 'ok' (verde) o 'aviso' (ámbar)."""
    color = _TONOS.get(tono, _TONOS["normal"])
    nota_html = f'<div class="ui-kpi-nota">{escape(nota)}</div>' if nota else ""
    _html(f'<div class="ui-kpi"><div class="ui-kpi-etiqueta">{escape(etiqueta)}</div>'
          f'<div class="ui-kpi-valor" style="color:{color}">{escape(valor)}</div>{nota_html}</div>', contenedor)


def fila_kpis(kpis: list) -> None:
    """Fila horizontal de tarjetas. Cada elemento: (etiqueta, valor) o (etiqueta, valor, nota[, tono])."""
    columnas = st.columns(len(kpis))
    for col, kpi in zip(columnas, kpis):
        etiqueta, valor, *resto = kpi
        nota = resto[0] if resto else ""
        tono = resto[1] if len(resto) > 1 else "normal"
        tarjeta_kpi(etiqueta, valor, nota, tono, col)


def tarjeta_lista(titulo: str, puntos: list, contenedor=None) -> None:
    """Tarjeta con título y viñetas. Los puntos pueden traer <b> y <code> (texto propio del dashboard)."""
    items = "".join(f"<li>{p}</li>" for p in puntos)
    _html(f'<div class="ui-tarjeta"><h4>{escape(titulo)}</h4><ul>{items}</ul></div>', contenedor)


def nota(texto: str, tipo: str = "") -> None:
    """Caja de interpretación. tipo: '' (neutra), 'aviso', 'ok' o 'alerta'. Acepta <b>."""
    _html(f'<div class="ui-nota {escape(tipo)}">{texto}</div>')


def pie(fuentes: list, equipo: str = "Equipo 1 · TI3008C · Tecnológico de Monterrey") -> None:
    """Pie de página común con equipo y fuentes de los datos."""
    _html(f'<div class="ui-pie"><span>{escape(equipo)}</span>'
          f'<span>Fuentes: {escape(" · ".join(fuentes))}</span></div>')
