"""Inserta controles B2 en el app.py EXISTENTE, sin eliminar su dashboard.

Usar una sola vez desde la raíz del proyecto:
    python scripts/integrar_app.py
Guarda una copia app.py.bak.b2 (no subirla a GitHub).
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"


def reemplazar_una_vez(texto: str, antes: str, despues: str) -> str:
    cuantos = texto.count(antes)
    if cuantos != 1:
        raise RuntimeError(f"No se reconoció el app.py actual (ancla encontrada {cuantos} veces): {antes[:85]!r}. No se modificó nada.")
    return texto.replace(antes, despues, 1)


def parchear(texto: str) -> str:
    if "# INICIO VALIDACION B2" in texto:
        return texto  # Idempotente: no duplica secciones.
    texto = reemplazar_una_vez(texto, "import streamlit as st\n", "import streamlit as st\nfrom validacion_datos import validar_dataset, resumen_segmentos\n")

    # El CSV se valida ANTES de todo dibujo, entrenamiento y KPIs.
    ancla = "bundle = None\nif MODEL_FILE.exists():"
    reemplazo = '''# INICIO VALIDACION B2: fail closed antes de mostrar cualquier cifra.
estado_calidad = None
if df is not None:
    estado_calidad = validar_dataset(df)
    df = estado_calidad.datos
    if not estado_calidad.valido:
        st.error("CONTRATO DE DATOS ROTO — Se detuvo el tablero para evitar cifras incorrectas.")
        for error in estado_calidad.errores:
            st.error(error)
        st.dataframe(estado_calidad.detalle, column_order=[
            "Columna", "Tipo esperado", "Tipo encontrado", "Nulos", "Estado"
        ], hide_index=True, use_container_width=True)
        st.stop()
# FIN VALIDACION B2

bundle = None
if MODEL_FILE.exists():'''
    texto = reemplazar_una_vez(texto, ancla, reemplazo)
    texto = reemplazar_una_vez(
        texto,
        '"🏠 Resumen ejecutivo", "📊 Explorar clientes", "🤖 Comparar modelos", "🎯 Predecir abandono"',
        '"🏠 Resumen ejecutivo", "📊 Explorar clientes", "🤖 Comparar modelos", "🧪 Calidad del dato", "🎯 Predecir abandono"',
    )
    ancla = 'st.caption("Clasificación de abandono de suscriptores | churned = 1 significa que cancela")'
    reemplazo = '''st.caption("Clasificación de abandono de suscriptores | churned = 1 significa que cancela")
# Resultado visible de la comprobación en todas las pantallas con datos.
if estado_calidad is not None:
    st.success(f"DATOS VALIDADOS · {len(estado_calidad.datos):,} filas · "
               "14 columnas fuente · números con tipo correcto")
    with st.expander("Ver verificación automática de columnas"):
        st.dataframe(estado_calidad.detalle, column_order=[
            "Columna", "Tipo esperado", "Tipo encontrado", "Nulos", "Estado"
        ], hide_index=True, use_container_width=True)
        for advertencia in estado_calidad.advertencias:
            st.warning(advertencia)
else:
    st.warning("SIN CSV VALIDADO: los indicadores del resumen y las métricas de "
               "modelos son resultados REPORTADOS del notebook, no datos recién verificados.")'''
    texto = reemplazar_una_vez(texto, ancla, reemplazo)

    ancla = '''else:
    st.title("Simulador de riesgo de abandono")'''
    reemplazo = '''elif pagina == "🧪 Calidad del dato":
    st.title("Calidad y procedencia del dato")
    st.caption("DATO = fuente CSV · CALCULADO = operación sobre el CSV · "
               "SIMULADO = entrada ficticia para explicar una decisión")
    if estado_calidad is None:
        st.error("Se necesita el CSV original para publicar cifras verificadas.")
        st.stop()
    st.subheader("Contrato ejecutable, revisado antes de dibujar")
    st.dataframe(estado_calidad.detalle, column_order=[
        "Columna", "Tipo esperado", "Tipo encontrado", "Nulos", "Estado"
    ], hide_index=True, use_container_width=True)
    st.subheader("Tabla fuente con tipos reales y orden numérico")
    st.write("Haz clic en **Cuota mensual · DATO** para ordenar de menor a mayor. "
             "La cuota es `float`, no texto; se conserva la unidad monetaria sin "
             "asignarle una moneda que la fuente no confirma.")
    tabla_dato = df.rename(columns={
        "subscription_type": "Plan · DATO",
        "monthly_fee": "Cuota mensual · DATO",
        "watch_hours": "Horas vistas · DATO",
        "last_login_days": "Días sin acceso · DATO",
        "churned": "Cancelación (0/1) · DATO",
    })[["Plan · DATO", "Cuota mensual · DATO", "Horas vistas · DATO",
        "Días sin acceso · DATO", "Cancelación (0/1) · DATO"]].copy()
    tabla_dato["Fuente"] = "CSV original: netflix_customer_churn.csv"
    st.dataframe(
        tabla_dato,
        column_order=["Plan · DATO", "Cuota mensual · DATO", "Horas vistas · DATO",
                      "Días sin acceso · DATO", "Cancelación (0/1) · DATO", "Fuente"],
        hide_index=True, use_container_width=True,
        column_config={"Cuota mensual · DATO": st.column_config.NumberColumn(
            "Cuota mensual · DATO", format="%.2f"),
                       "Horas vistas · DATO": st.column_config.NumberColumn(
                           "Horas vistas · DATO", format="%.2f")},
    )
    st.subheader("Indicadores calculados sobre datos reales")
    segmentos = resumen_segmentos(df).rename(columns={
        "Clientes": "Clientes · CALCULADO",
        "Abandonos": "Abandonos · CALCULADO",
        "Tasa de abandono (%)": "Abandono (%) · CALCULADO",
    })
    segmentos["Origen / fórmula"] = "Conteo y tasa sobre churned del CSV"
    st.dataframe(segmentos, column_order=[
        "Plan", "Clientes · CALCULADO", "Abandonos · CALCULADO",
        "Abandono (%) · CALCULADO", "Origen / fórmula"
    ], hide_index=True, use_container_width=True,
       column_config={"Abandono (%) · CALCULADO": st.column_config.NumberColumn(
           "Abandono (%) · CALCULADO", format="%.2f")})
    st.caption("Tasa de abandono (%) = 100 × abandonos / clientes en cada plan.")
    if st.checkbox("Mostrar una demostración de cifra SIMULADA (no es un resultado real)"):
        st.warning("DEMO / SIMULADO: suponemos 100 contactos y 30 respuestas para "
                   "ilustrar un cálculo hipotético. No provienen de Netflix, "
                   "SAS, la muestra ni de un experimento de retención.")
        st.metric("Respuestas hipotéticas · SIMULADO", "30 de 100")

else:
    st.title("Simulador de riesgo de abandono")
    st.info("SIMULADO / ESTIMADO: este perfil lo introduce el usuario; "
            "no equivale a una observación real ni demuestra impacto de una campaña.")'''
    texto = reemplazar_una_vez(texto, ancla, reemplazo)
    # Etiquetar la tabla principal de exploración, sin perder el orden numérico.
    texto = reemplazar_una_vez(texto,
        '''        st.dataframe(seg.rename(columns={opciones[segmento]: "Segmento", "clientes": "Clientes"})
                     [["Segmento", "Clientes", "Abandono (%)"]].round(1),
                     hide_index=True, use_container_width=True)''',
        '''        tabla_segmento = (seg.rename(columns={opciones[segmento]: "Segmento", "clientes": "Clientes"})
                          [["Segmento", "Clientes", "Abandono (%)"]].round(1))
        tabla_segmento = tabla_segmento.rename(columns={
            "Segmento": "Segmento · DATO", "Clientes": "Clientes · CALCULADO",
            "Abandono (%)": "Abandono (%) · CALCULADO"})
        tabla_segmento["Origen"] = "Fuente: CSV · tasa = 100 × media(churned)"
        st.dataframe(tabla_segmento,
                     column_order=["Segmento · DATO", "Clientes · CALCULADO",
                                   "Abandono (%) · CALCULADO", "Origen"],
                     hide_index=True, use_container_width=True)''')
    # Etiquetar una segunda tabla: estas métricas se reportan desde el notebook.
    texto = reemplazar_una_vez(texto,
        '''        st.dataframe(RESULTADOS_TEST.set_index("Modelo")
                     .sort_values("F1", ascending=False).style.format("{:.2%}"),
                     use_container_width=True)''',
        '''        tabla_modelos = RESULTADOS_TEST.sort_values("F1", ascending=False).copy()
        tabla_modelos["Origen"] = "REPORTADO · notebook, prueba"
        st.dataframe(tabla_modelos.style.format({c: "{:.2%}" for c in
                     ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]}),
                     column_order=["Modelo", "Accuracy", "Precision", "Recall", "F1",
                                   "ROC-AUC", "Origen"],
                     hide_index=True, use_container_width=True)''')
    texto = reemplazar_una_vez(texto,
                               'a.metric("Probabilidad estimada de abandono", f"{prob:.1%}")',
                               'a.metric("Probabilidad estimada · SIMULADA", f"{prob:.1%}")')
    # Errores al abrir o subir CSV no deben transformarse en indicadores de respaldo invisibles.
    texto = reemplazar_una_vez(texto,
                               'st.error(f"No fue posible leer el CSV: {exc}")',
                               'st.error(f"No fue posible leer el CSV: {exc}")\n        st.stop()')
    texto = reemplazar_una_vez(texto,
                               'st.sidebar.error(f"CSV inválido: {exc}")',
                               'st.sidebar.error(f"CSV inválido: {exc}")\n            st.stop()')
    return texto


def main() -> None:
    if not APP.is_file():
        raise SystemExit(f"No existe {APP}. Copia estos archivos a la raíz correcta del repo.")
    original = APP.read_text(encoding="utf-8")
    actualizado = parchear(original)
    if original == actualizado:
        print("La integración B2 ya estaba aplicada; no se duplicó.")
        return
    copia = APP.with_name("app.py.bak.b2")
    copia.write_text(original, encoding="utf-8")
    # Comprobamos sintaxis antes de sustituir el dashboard.
    compile(actualizado, str(APP), "exec")
    APP.write_text(actualizado, encoding="utf-8")
    print("Integración B2 aplicada en app.py; respaldo local en app.py.bak.b2.")
    print("Ahora ejecuta: python scripts/validar_contrato.py")
    print("Y: streamlit run app.py")


if __name__ == "__main__":
    main()
