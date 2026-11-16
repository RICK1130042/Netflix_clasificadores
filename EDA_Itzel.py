from pathlib import Path

import pandas as pd
from streamlit.runtime.scriptrunner import get_script_run_ctx
import plotly.express as px
import streamlit as st


CARPETA_SCRIPT = Path(__file__).resolve().parent
UBICACIONES_CSV = [
    CARPETA_SCRIPT / "netflix_customer_churn.csv",
    CARPETA_SCRIPT.parent / "netflix_customer_churn.csv",
]
ARCHIVO_CSV = next((ruta for ruta in UBICACIONES_CSV if ruta.exists()), None)
COLUMNAS_NECESARIAS = [
    "subscription_type", "watch_hours", "last_login_days", "churned", "favorite_genre"
]
COLORES = {"Permanece": "#7F8C9A", "Canceló": "#E50914"}

if get_script_run_ctx(suppress_warning=True) is None:
    raise SystemExit(
        "Esta es una app de Streamlit. Ejecútala en Terminal con:\n"
        "python3 -m streamlit run ~/Downloads/EDA_Itzel.py"
    )

st.set_page_config(page_title="EDA de Netflix | Plotly", page_icon="📊", layout="wide")
st.title("Exploración de usuarios de Netflix")
st.write(
    "Esta página permite explorar el comportamiento de los usuarios y "
    "comparar suscripciones, visualización, inactividad y cancelaciones."
)


def validar_datos(datos: pd.DataFrame) -> pd.DataFrame:
    faltantes = [col for col in COLUMNAS_NECESARIAS if col not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas necesarias: {', '.join(faltantes)}")

    datos = datos.copy()
    for col in ["watch_hours", "last_login_days", "churned"]:
        entrada = datos[col].str.strip()
        convertido = pd.to_numeric(entrada, errors="coerce")
        if convertido.isna().any():
            raise ValueError(f"La columna '{col}' contiene valores vacíos o que no son números.")
        datos[col] = convertido

    if not datos["churned"].isin([0, 1]).all():
        raise ValueError("La columna 'churned' debe contener solamente 0 o 1.")
    if (datos["watch_hours"] < 0).any() or (datos["last_login_days"] < 0).any():
        raise ValueError("Las horas de visualización y los días sin acceso no pueden ser negativos.")
    if datos["subscription_type"].str.strip().eq("").any():
        raise ValueError("Hay registros sin tipo de suscripción.")

    datos["favorite_genre"] = datos["favorite_genre"].replace("", "No registrado")
    datos["Estado"] = datos["churned"].map({0: "Permanece", 1: "Canceló"})
    return datos


@st.cache_data(show_spinner=False)
def cargar_datos(path: str, ultima_modificacion: int) -> pd.DataFrame:
    datos = pd.read_csv(path, dtype=str, keep_default_na=False)
    return validar_datos(datos)

try:
    if ARCHIVO_CSV is not None:
        df = cargar_datos(str(ARCHIVO_CSV), ARCHIVO_CSV.stat().st_mtime_ns)
        st.caption(f"Archivo cargado: {ARCHIVO_CSV.name}")
    else:
        archivo_subido = st.file_uploader(
            "Selecciona el archivo netflix_customer_churn.csv para ver las gráficas",
            type=["csv"],
        )
        if archivo_subido is None:
            st.info("Puedes subir aquí el CSV o guardarlo junto a EDA_Itzel.py.")
            st.stop()
        df = validar_datos(pd.read_csv(archivo_subido, dtype=str, keep_default_na=False))
except (ValueError, OSError, pd.errors.ParserError) as error:
    st.error(f"No se pueden mostrar las gráficas porque los datos no pasaron la revisión: {error}")
    st.stop()

st.success("Datos verificados: las columnas necesarias tienen los tipos esperados.")
planes = sorted(df["subscription_type"].unique())
seleccion = st.multiselect("Filtrar por tipo de suscripción", options=planes, default=planes)
filtrado = df[df["subscription_type"].isin(seleccion)].copy()
if filtrado.empty:
    st.info("Selecciona al menos un plan para mostrar los resultados.")
    st.stop()

clientes = len(filtrado)
abandonos = int(filtrado["churned"].sum())
col1, col2, col3 = st.columns(3)
col1.metric("Usuarios incluidos · CALCULADO", f"{clientes:,}")
col2.metric("Cancelaciones registradas · CALCULADO", f"{abandonos:,}")
col3.metric("Tasa de abandono · CALCULADO", f"{filtrado['churned'].mean():.1%}")
st.caption("Fuente: netflix_customer_churn.csv. Los conteos y porcentajes son cálculos sobre los datos filtrados.")

tasas_plan = (
    filtrado.groupby("subscription_type", as_index=False)
    .agg(Usuarios=("churned", "size"), Tasa=("churned", "mean"))
)
tasas_plan["Abandono (%)"] = tasas_plan["Tasa"] * 100
fig1 = px.bar(
    tasas_plan, x="subscription_type", y="Abandono (%)", text="Abandono (%)",
    hover_data={"Usuarios": True, "Tasa": False},
    labels={"subscription_type": "Tipo de suscripción"},
    title="1. Tasa de abandono según el plan", color_discrete_sequence=["#E50914"]
)
fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig1.update_layout(yaxis_range=[0, max(100, tasas_plan["Abandono (%)"].max() * 1.1)])
st.plotly_chart(fig1, use_container_width=True)
st.caption("CALCULADO: porcentaje de usuarios que cancelaron dentro de cada plan.")

fig2 = px.box(
    filtrado, x="Estado", y="watch_hours", color="Estado", points=False,
    color_discrete_map=COLORES,
    labels={"watch_hours": "Horas de visualización · DATO", "Estado": "Estado del usuario"},
    title="2. Horas de visualización según cancelación"
)
fig2.update_layout(showlegend=False)
st.plotly_chart(fig2, use_container_width=True)
st.caption("DATO: las horas vienen del CSV. Su periodo de referencia no está confirmado en el archivo.")

fig3 = px.histogram(
    filtrado, x="last_login_days", color="Estado", nbins=25, barmode="overlay",
    opacity=0.72, color_discrete_map=COLORES,
    labels={"last_login_days": "Días desde el último acceso · DATO", "count": "Usuarios"},
    title="3. Inactividad de los usuarios"
)
fig3.update_layout(yaxis_title="Número de usuarios", legend_title_text="Estado")
st.plotly_chart(fig3, use_container_width=True)
st.caption("DATO: días sin acceso registrados. El número de usuarios en cada intervalo es calculado.")

tasas_genero = (
    filtrado.groupby("favorite_genre", as_index=False)
    .agg(Usuarios=("churned", "size"), Tasa=("churned", "mean"))
)
tasas_genero["Abandono (%)"] = tasas_genero["Tasa"] * 100
tasas_genero = tasas_genero.sort_values("Abandono (%)", ascending=True)
fig4 = px.bar(
    tasas_genero, x="Abandono (%)", y="favorite_genre", orientation="h",
    hover_data={"Usuarios": True, "Tasa": False},
    labels={"favorite_genre": "Género favorito"},
    title="4. Tasa de abandono según género favorito", color_discrete_sequence=["#E50914"]
)
fig4.update_layout(xaxis_range=[0, max(100, tasas_genero["Abandono (%)"].max() * 1.1)])
st.plotly_chart(fig4, use_container_width=True)
st.caption("CALCULADO: tasa por género favorito. Al interpretar diferencias, revisa también el tamaño de cada grupo.")

st.info("Estos gráficos describen asociaciones en la base disponible; por sí solos no explican las causas de la cancelación.")
