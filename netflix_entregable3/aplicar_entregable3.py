"""Aplicar desde la carpeta del repo: python /ruta/aplicar_entregable3.py
Prepara cambios en memoria, verifica sintaxis y escribe solo si la revisión coincide.
No hace commits, push, ni configura secretos.
"""
from pathlib import Path
import ast
import shutil
import sys

BASE = Path(__file__).resolve().parent
ROOT = Path.cwd()
APP = ROOT / 'app.py'
UI = ROOT / 'ui.py'

QUESTIONS = {
315: '¿A partir de qué número de clústeres disminuye la mejora de la compactación de grupos?',
348: '¿Cómo cambia la discriminación del abandono al usar K = 6 frente a otro K?',
349: '¿Qué cantidad de clústeres merece validación adicional por su AUC exploratorio?',
372: '¿Qué tamaño y tasa de abandono tiene cada grupo para orientar la capacidad de retención?',
533: '¿Qué modelo se debe seleccionar por F1 en validación cruzada?',
642: '¿Qué segmento presenta mayor tasa de abandono y merece atención prioritaria?',
671: '¿Qué diferencias de horas vistas existen entre quienes permanecen y quienes cancelan?',
692: '¿Qué variables se asocian más con el abandono y merecen seguimiento?',
723: '¿Cómo se comparan los modelos en prueba según la métrica seleccionada?',
748: '¿Qué variables aportan más al desempeño del modelo y conviene vigilar?',
789: '¿Cuántos falsos negativos y falsos positivos produce el modelo seleccionado?',
805: '¿Qué equilibrio entre detección y falsas alarmas ofrece el modelo al variar el umbral?',
812: '¿Qué equilibrio entre precisión y recall ofrece el modelo al variar el umbral?',
949: '¿Cuántos clientes requieren cada nivel de atención para dimensionar la campaña?',
961: '¿En qué plan hay más clientes de prioridad alta para asignar recursos?',
1056: '¿Cómo se distribuyen los scores por prioridad para revisar la cartera?',
1194: '¿Qué riesgo estima el modelo para el perfil y qué prioridad corresponde?',
1242: '¿Qué variables aumentan o reducen la predicción de abandono?',
1278: '¿Cómo cambia el riesgo estimado si varían los días sin acceso?',
1281: '¿Cómo cambia el riesgo estimado si varían las horas vistas?',
}


def main():
    if not APP.is_file() or not UI.is_file() or not (ROOT / 'README.md').is_file():
        sys.exit('Abre la terminal en la raíz del repositorio Netflix_clasificadores.')
    original = APP.read_text(encoding='utf-8').splitlines()
    if len(original) not in (1291, 1292) or 'ROJO = "#E50914"' not in original[60]:
        sys.exit('La versión de app.py no coincide con la revisión auditada. No se modificó nada.')
    edits = []
    def change(start, end, marker, new):
        if marker not in original[start-1]:
            sys.exit(f'No coincide app.py línea {start}: {original[start-1]!r}. No se modificó nada.')
        edits.append((start, end, new))
    def insert(before, marker, new):
        change(before, before, marker, new + '\n' + original[before-1])
    # Importaciones y control de acceso antes de cargar información.
    insert(30, 'from sklearn.cluster', 'from palette import COLORES\nimport auth')
    change(61, 63, 'ROJO =', 'ROJO = COLORES["rojo"]\nGRIS = COLORES["gris"]\nVERDE = COLORES["verde"]\nAMBAR = COLORES["ambar"]\nAZUL = COLORES["azul"]')
    change(79, 79, 'ui.aplicar_estilos()', 'ui.aplicar_estilos()\nrol = auth.iniciar_sesion()')
    insert(182, 'colores =', '''    if tabla.empty:
        fig = go.Figure()
        fig.add_annotation(text="No hay valores suficientes para esta gráfica.", showarrow=False)
        return fig''')
    # K-Means: una comparación elegida, con trazos diferenciados.
    change(327, 334, 'colores =', '''        k_otro = st.selectbox("Comparar K = 6 con", [k for k in VALORES_K if k != K_SELECCIONADO], key="km_comparador")
        curva = curvas[k_otro]
        roc.add_trace(go.Scatter(x=curva["fpr"], y=curva["tpr"], mode="lines",
                                 name=f'K = {k_otro} | AUC = {curva["auc"]:.4f}',
                                 line=dict(color=AZUL, width=3, dash="dot")))''')
    change(362, 362, 'font-weight: bold; color:', '                     .apply(lambda fila: [f"font-weight: bold; color: {ROJO}"')
    change(367, 371, 'fig = px.bar(segmentos', '''        fig = px.bar(segmentos, x="Cluster", y="Clientes", text="Clientes",
                     color_discrete_sequence=[AZUL], hover_data={"Abandono": ":.1%"})
        fig.update_xaxes(tickmode="array", tickvals=list(range(1, K_SELECCIONADO + 1)))
        fig.update_layout(height=350)''')
    # No permitir que gerencia suba archivos con datos individuales.
    change(395, 395, 'archivo = st.sidebar.file_uploader', '    archivo = st.sidebar.file_uploader("Cargar dataset CSV", type="csv") if rol == "analista" else None')
    change(440, 443, 'pagina = st.sidebar.radio', '''pagina = st.sidebar.radio("Navegación", auth.PAGINAS[rol], label_visibility="collapsed", key="navegacion")
auth.exigir_pagina(rol, pagina)''')
    change(465, 473, 'if elegido not in', '''    if elegido is None:
        st.info("No seleccionaste ningún segmento. Elige un clúster o Todos.")
        st.caption("Clientes seleccionados: 0")
        st.stop()
    if elegido != 0:
        df_filtrado = df_segmentado.loc[df_segmentado["cluster"] == elegido].copy()
        st.markdown(f"**Perfil:** {nombres[elegido]}")
    filtro_cluster = "Todos" if elegido == 0 else f"Clúster {elegido}"
    st.caption(f"Segmento activo: {filtro_cluster} · {len(df_filtrado):,} clientes de {len(df_segmentado):,}")
    if df_filtrado.empty:
        st.info("No hay clientes en el segmento seleccionado.")
        st.stop()
    st.caption("Segmentos según etiquetas_clusters.csv y perfiles_clusters.csv.")''')
    change(622, 625, 'sub = df_filtrado', '''    sub = df_filtrado[df_filtrado["subscription_type"].isin(planes)].copy()
    filtro_cluster = "Todos" if elegido == 0 else f"Clúster {elegido}"
    st.caption(f"Planes activos: {', '.join(planes) or 'Ninguno'} · Segmento: {filtro_cluster} · Agrupación: {segmento}")
    st.caption(f"Clientes filtrados: {len(sub):,} de {len(df):,}")
    if sub.empty:
        st.info("No hay clientes con este filtro. Elige al menos un plan con registros.")
        st.stop()''')
    insert(637, 'seg[col] =', '    if seg.empty:\n        st.info("La agrupación no contiene categorías válidas.")\n        st.stop()')
    change(656, 658, 'st.download_button', '''        if rol == "analista":
            st.download_button("Descargar clientes filtrados (CSV)",
                               sub.to_csv(index=False).encode("utf-8"),
                               file_name="clientes_filtrados.csv", mime="text/csv", width="stretch")
        else:
            st.caption("La exportación de registros individuales está reservada al analista.")''')
    change(664, 665, 'fig = px.box', '''        fig = px.box(caja, x="Estado", y="watch_hours", points=False, color="Estado",
                     color_discrete_map={"0 · Permanece": AZUL, "1 · Cancela": ROJO},''')
    change(676, 678, 'st.caption(f"Puntos', '''        st.caption(f"Atípicos (regla 1.5 × RIC): {atipicos.get(0, 0)} en quienes permanecen "
                   f"y {atipicos.get(1, 0)} en quienes cancelan. No se muestran puntos individuales.")''')
    # Evitar texto sobre gradientes sin contraste garantizado: barras y tabla numérica.
    change(687, 691, 'fig = px.imshow', '''        rel_grafica = correlacion["churned"].drop("churned").dropna().sort_values(key=abs)
        tabla_corr = pd.DataFrame({"Variable": [cortos.get(c, c) for c in rel_grafica.index], "rho": rel_grafica.values})
        fig = grafico_barras(tabla_corr, "rho", "Variable", "Correlación de Spearman con abandono", formato="dec", eje="ρ")''')
    change(781, 781, 't1, t2, t3 =', '            t1, t2 = st.tabs(["Matriz de confusión", "Curvas ROC y PR"])')
    change(784, 788, 'fig = px.imshow', '''                etiquetas_cm = ["VN · Permanece detectado", "FP · Falsa alerta", "FN · Abandono no detectado", "VP · Abandono detectado"]
                fig = go.Figure(go.Bar(x=cm.ravel(), y=etiquetas_cm, orientation="h",
                                       marker_color=[AZUL, ROJO, ROJO, AZUL], text=cm.ravel(), textposition="outside"))
                fig.update_layout(title=f"Matriz de confusión (conteos) · {modelo_nombre}", xaxis_title="Clientes", height=400)''')
    change(813, 820, 'with t3:', '            # El árbol no se conserva: no evalúa necesariamente el clasificador seleccionado.')
    # Gerencia ve sus indicadores, no las pestañas que ejecutan consultas CRM.
    change(927, 929, 'laura, andres, sofia, medicion =', '''    if rol == "gerencia":
        laura = st.container()
    else:
        laura, andres, sofia, medicion = st.tabs([
            "Laura · Gerencia", "Andrés · Campañas CRM", "Sofía · Calidad", "Medición · Prueba A/B"
        ])''')
    insert(995, 'with andres:', '''    if rol != "analista":
        st.caption("CRM, registros individuales y asignación A/B requieren el rol analista.")
        ui.pie(["Dataset del proyecto", "Resultados del notebook"])
        st.stop()''')
    insert(1018, 'k1, k2, k3 =', '''        st.caption(f"Campañas: {', '.join(tipos_camp) or 'Ninguna'} · Planes: {', '.join(planes_camp) or 'Ninguno'} · Segmentos: {', '.join(segmentos_camp) or 'Ninguno'} · Solo zona crítica: {'Sí' if solo_critica else 'No'}")
        if audiencia.empty:
            st.info("No hay clientes con los filtros seleccionados. Ajusta la audiencia.")
            st.caption("Audiencia seleccionada: 0 clientes")
            st.stop()''')
    change(1052, 1054, 'fig = px.histogram', '''        fig = px.histogram(camp, x="Score", facet_col="Prioridad", nbins=25,
                           color_discrete_sequence=[AZUL],
                           title="Distribución de puntuaciones por prioridad (muestra histórica)")''')
    # Riesgo: barra neutral y texto, sin franjas que dependan del color.
    change(1190, 1192, '"steps":', '''                    "bgcolor": COLORES["tarjeta"], "bordercolor": COLORES["borde"], "borderwidth": 1,
                    "threshold": {"line": {"color": COLORES["blanco"], "width": 3}, "value": 50}}))''')
    change(1195, 1195, 'st.caption("Franjas:', '        st.caption("Bajo < 30 %, medio 30–59 %, alto ≥ 60 %. Línea blanca: umbral de 50 % del modelo.")')
    change(1263, 1265, 'for y0, y1, c in', '''        for limite_riesgo in (30, 60):
            fig.add_hline(y=limite_riesgo, line_dash="dot", line_color=GRIS,
                          annotation_text=f"Umbral {limite_riesgo} %")''')
    # Agrega preguntas explícitas antes de cada llamada que pinta una gráfica.
    for n, pregunta in QUESTIONS.items():
        line = original[n-1]
        if 'st.plotly_chart(' not in line:
            sys.exit(f'No coincide gráfica en línea {n}. No se modificó nada.')
        indent = line[:len(line)-len(line.lstrip())]
        new = indent + f'ui.grafica({pregunta!r}, ' + line.lstrip().removeprefix('st.plotly_chart(')
        change(n, n, 'st.plotly_chart(', new)
    # Comprobar que no haya ediciones solapadas y aplicar desde abajo.
    for a, b in zip(sorted(edits), sorted(edits)[1:]):
        if a[1] >= b[0]:
            sys.exit(f'Ediciones solapadas: {a[0]} / {b[0]}')
    lines = original[:]
    for start, end, new in sorted(edits, reverse=True):
        lines[start-1:end] = new.splitlines()
    app = '\n'.join(lines) + '\n'
    app = app.replace('"#F5A524"', 'AMBAR').replace('"#777"', 'COLORES["borde"]').replace('color="white"', 'color=COLORES["blanco"]')
    # Garantizar el color de texto y que la categoría también aparezca escrita.
    ui = UI.read_text(encoding='utf-8')
    start = ui.index('COLORES = {')
    end = ui.index('\n_TONOS', start)
    ui = ui[:start] + 'from palette import COLORES\n' + ui[end:]
    ui = ui.replace('#D5D9E0', 'var(--texto)').replace('background: rgba(255,255,255,.04)', 'background: var(--tarjeta)')
    ui = ui.replace('</style>', '''
/* Botones primarios: texto oscuro sobre el acento claro. */
button[kind="primary"], button[kind="primary"] * {{color: {COLORES['negro']} !important;}}
</style>''', 1)
    ui += '''\n\ndef grafica(pregunta, figura, **kwargs):
    """Pregunta visible y fondo fijo; evita colores de tema no comprobados."""
    st.markdown(f"**Pregunta:** {pregunta}")
    figura.update_layout(paper_bgcolor=COLORES["fondo"], plot_bgcolor=COLORES["fondo"],
                         font=dict(color=COLORES["texto"]),
                         colorway=[COLORES["azul"], COLORES["rojo"], COLORES["ambar"], COLORES["gris"]])
    figura.update_xaxes(gridcolor=COLORES["borde"], zerolinecolor=COLORES["borde"])
    figura.update_yaxes(gridcolor=COLORES["borde"], zerolinecolor=COLORES["borde"])
    for traza in figura.data:
        if traza.type == "bar":
            traza.update(textposition="outside", textfont=dict(color=COLORES["texto"]), cliponaxis=False)
    st.plotly_chart(figura, theme=None, **kwargs)
'''
    for name, source in [('app.py', app), ('ui.py', ui)]:
        ast.parse(source, filename=name)
    # No sobrescribir un entregable existente por accidente.
    paths = ['palette.py', 'auth.py', '.streamlit/config.toml', 'tests/test_contrast.py',
             '.github/workflows/contrast.yml', 'docs/entregable-3.md', 'prompts/B3.md']
    existing = [p for p in paths if (ROOT / p).exists()]
    if existing:
        sys.exit(f'Estos archivos ya existen: {existing}. Revisa y fusiona manualmente; no se modificó nada.')
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    link = '[Dashboard en Streamlit Cloud](https://rick1130042-netflix-clasificadores-app-rzfx7h.streamlit.app/)'
    if not readme.startswith(link):
        readme = link + '\n\n' + readme
    ignore = ROOT / '.gitignore'
    ig = ignore.read_text(encoding='utf-8') if ignore.exists() else ''
    for pattern in ('.streamlit/secrets.toml', '.env', '.env.*', '__pycache__/', '*.pyc', '.DS_Store', '.venv/'):
        if pattern not in ig.splitlines():
            ig += '\n' + pattern + '\n'
    for p in paths:
        target = ROOT / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BASE / p, target)
    APP.write_text(app, encoding='utf-8')
    UI.write_text(ui, encoding='utf-8')
    (ROOT / 'README.md').write_text(readme, encoding='utf-8')
    ignore.write_text(ig, encoding='utf-8')
    print('Cambios preparados. Revisa git diff, configura Secrets, ejecuta pruebas y prueba ambos roles. No se hizo commit.')


if __name__ == '__main__':
    main()
