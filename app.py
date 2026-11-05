"""Netflix | Churn Analytics — Dashboard Streamlit.

Basado en el notebook netflix_churned.html.
Coloca este archivo junto a netflix_customer_churn.csv y, opcionalmente,
modelo_churn_netflix.joblib (exportado en la última celda del notebook).
Ejecuta: streamlit run app.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


# ─── Configuración del proyecto ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
CSV_FILE = ROOT / "netflix_customer_churn.csv"
MODEL_FILE = ROOT / "modelo_churn_netflix.joblib"
NUMERICAS = [
    "watch_hours", "last_login_days", "avg_watch_time_per_day", "number_of_profiles"
]
PLANES = ["Basic", "Premium", "Standard"]
COLUMNAS = NUMERICAS + [f"subscription_type_{plan}" for plan in PLANES]
MODELOS = ["Regresión Logística", "Árbol de Decisión", "Random Forest", "Gradient Boosting"]
ROJO = "#E50914"
GRIS = "#9AA0AA"

# Cifras transcritas de la evaluación en PRUEBA del notebook (N=1,000).
RESULTADOS_TEST = pd.DataFrame([
    ["Regresión Logística", .8840, .8700, .9046, .8869, .9583],
    ["Árbol de Decisión", .9350, .9720, .8966, .9328, .9843],
    ["Random Forest", .9360, .9545, .9165, .9351, .9898],
    ["Gradient Boosting", .9300, .9391, .9205, .9297, .9903],
], columns=["Modelo", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])

# Resultados de validación cruzada (cinco folds, escenario: todas las seleccionadas).
RESULTADOS_CV = pd.DataFrame([
    ["Regresión Logística", .8838, .8872],
    ["Árbol de Decisión", .9347, .9071],
    ["Random Forest", .9352, .9140],
    ["Gradient Boosting", .9392, .9215],
], columns=["Modelo", "F1", "Recall"])

ESCENARIOS_GB = pd.DataFrame([
    ["Todas las seleccionadas", .9392],
    ["Sin avg_watch_time_per_day", .9323],
    ["Sin last_login_days", .9249],
    ["Sin ambas", .7390],
], columns=["Escenario", "F1 CV"])

IMPORTANCIAS = pd.DataFrame([
    ["avg_watch_time_per_day", .1795],
    ["watch_hours", .1336],
    ["last_login_days", .1155],
    ["subscription_type_Basic", .0578],
    ["number_of_profiles", .0518],
    ["subscription_type_Standard", .0000],
    ["subscription_type_Premium", -.0009],
], columns=["Variable", "Disminución del F1"])


st.set_page_config(
    page_title="Netflix · Churn Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
    h1, h2, h3 {letter-spacing: -.025em;}
    div[data-testid="stMetric"] {background: #1B1F27; padding: 16px 20px;
        border: 1px solid #313640; border-radius: 12px;}
    div[data-testid="stMetricValue"] {font-size: 1.9rem;}
    div[data-testid="stMetric"] * {color: #F5F6FA !important;}
    .block-container {padding-top: 1.5rem; max-width: 1300px;}
    .netflix-brand {font-weight: 900; font-size: 1.4rem; color: #E50914;
        letter-spacing: .12em;}
    .small-note {color: #B9BEC8; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def leer_csv(path: str, version: int) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def leer_modelo(path: str, version: int) -> dict:
    # Abrir SOLO un .joblib propio generado por el notebook.
    return joblib.load(path)


def preparar_x(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    necesarias = NUMERICAS + ["subscription_type", "churned"]
    faltantes = [col for col in necesarias if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {', '.join(faltantes)}")
    if df[necesarias].isna().any().any():
        raise ValueError("El dataset contiene nulos en las variables del modelo.")
    if not df["churned"].isin([0, 1]).all():
        raise ValueError("churned debe contener únicamente 0 y 1.")
    x = pd.get_dummies(
        df[NUMERICAS + ["subscription_type"]],
        columns=["subscription_type"], dtype=int
    ).reindex(columns=COLUMNAS, fill_value=0)
    y = df["churned"].astype(int)
    return x, y


def construir_modelos() -> dict:
    # Mismos hiperparámetros y preprocesamiento que en el notebook.
    preprocesador = ColumnTransformer(
        [("escalado", StandardScaler(), NUMERICAS)], remainder="passthrough"
    )
    return {
        "Regresión Logística": Pipeline([
            ("preprocesamiento", preprocesador),
            ("clasificador", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "Árbol de Decisión": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=10, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=9, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120, learning_rate=.06, max_depth=3, random_state=42
        ),
    }


@st.cache_resource(show_spinner="Preparando los clasificadores con el CSV...")
def entrenar_para_dashboard(df: pd.DataFrame) -> dict:
    """Reconstruye los 4 clasificadores sobre el split fijo del notebook.

    Se cachean modelos entrenados; NO se reentrenan al mover un filtro.
    No ejecuta la validación cruzada: sus resultados vienen del notebook.
    """
    x, y = preparar_x(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=.20, random_state=42, stratify=y
    )
    modelos = construir_modelos()
    for modelo in modelos.values():
        modelo.fit(x_train, y_train)
    return {"modelos": modelos, "x_test": x_test, "y_test": y_test}


def grafico_barras(tabla: pd.DataFrame, x: str, y: str, porcentaje=False):
    fig = px.bar(tabla, x=x, y=y, orientation="h", text=x,
                 color_discrete_sequence=[ROJO])
    fig.update_traces(texttemplate="%{x:.1f}%" if porcentaje else "%{x:.3f}",
                      textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=0, r=65, t=15, b=10), height=300,
                      yaxis_title=None, xaxis_title=x,
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ─── Dataset y recursos disponibles ─────────────────────────────────────────
df = None
if CSV_FILE.exists():
    try:
        df = leer_csv(str(CSV_FILE), CSV_FILE.stat().st_mtime_ns)
    except Exception as exc:
        st.error(f"No fue posible leer el CSV: {exc}")
else:
    st.sidebar.caption("No se encontró netflix_customer_churn.csv en esta carpeta.")
    archivo = st.sidebar.file_uploader("Cargar dataset CSV", type="csv")
    if archivo is not None:
        try:
            df = pd.read_csv(archivo)
        except Exception as exc:
            st.sidebar.error(f"CSV inválido: {exc}")

bundle = None
if MODEL_FILE.exists():
    try:
        bundle = leer_modelo(str(MODEL_FILE), MODEL_FILE.stat().st_mtime_ns)
    except Exception as exc:
        st.sidebar.warning(f"No se pudo cargar el modelo guardado: {exc}")

st.sidebar.markdown('<div class="netflix-brand">NETFLIX</div>', unsafe_allow_html=True)
st.sidebar.caption("CHURN ANALYTICS · MODELOS DE CLASIFICACIÓN")
pagina = st.sidebar.radio("Navegación", [
    "🏠 Resumen ejecutivo", "📊 Explorar clientes", "🤖 Comparar modelos", "🎯 Predecir abandono"
], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption(f"Dataset: {'disponible' if df is not None else 'sin cargar'}")
st.sidebar.caption(f"Modelo exportado: {'disponible' if bundle else 'sin cargar'}")
st.sidebar.caption("Fuente de métricas: notebook netflix_churned · prueba 80/20")

st.markdown('<div class="netflix-brand">NETFLIX <span style="color:#ECEFF3;'
            'font-size:1rem;font-weight:500;letter-spacing:0">/ CHURN ANALYTICS</span>'
            '</div>', unsafe_allow_html=True)
st.caption("Clasificación de abandono de suscriptores | churned = 1 significa que cancela")


# ─── 1. Resumen ─────────────────────────────────────────────────────────────
if pagina == "🏠 Resumen ejecutivo":
    st.title("Panorama del abandono")
    st.write("Una visión del problema de negocio, los clientes y los modelos entrenados.")
    total = len(df) if df is not None else 5000
    tasa = float(df["churned"].mean()) if df is not None and "churned" in df else .503
    a, b, c, d = st.columns(4)
    a.metric("Clientes analizados", f"{total:,}")
    b.metric("Tasa de abandono", f"{tasa:.1%}")
    c.metric("F1 del modelo elegido (prueba)", "92.97 %")
    d.metric("Recall del modelo elegido", "92.05 %")
    if df is None:
        st.info("Cifras descriptivas provenientes del notebook. Agrega el CSV para explorar segmentos.")
    st.subheader("¿Qué modelo se seleccionó?")
    izq, der = st.columns([1.5, 1])
    with izq:
        st.write("**Gradient Boosting** fue elegido mediante validación cruzada "
                 "(F1 promedio: **93.92 %**). En la prueba final alcanzó "
                 "un F1 de **92.97 %** y Recall de **92.05 %** para abandono.")
        st.write("**Random Forest** tuvo el mejor F1 en el conjunto de prueba "
                 "(**93.51 %**), pero no fue el ganador de la selección previa. "
                 "Esto evita elegir el modelo usando los datos de prueba.")
        st.success("Hipótesis cumplida: F1 y Recall ≥ 90 % en la clase churned = 1.")
    with der:
        graf = RESULTADOS_CV.sort_values("F1")
        st.plotly_chart(grafico_barras(graf, "F1", "Modelo"), use_container_width=True)
        st.caption("F1 promedio · validación cruzada (5 particiones)")
    st.divider()
    st.subheader("Patrones principales del análisis exploratorio")
    p, q = st.columns(2)
    p.metric("Mediana de horas vistas · permanece", "13.86 h")
    q.metric("Mediana de horas vistas · cancela", "4.06 h")
    st.caption("Los clientes que cancelan muestran menor visualización y más días "
               "desde su último acceso (medianas: 41 vs. 19 días). Asociación, no causalidad.")


# ─── 2. Exploración de clientes ──────────────────────────────────────────────
elif pagina == "📊 Explorar clientes":
    st.title("Explorar el comportamiento")
    if df is None:
        st.warning("Para mostrar gráficas interactivas carga netflix_customer_churn.csv.")
        st.stop()
    columnas_requeridas = ["churned", "subscription_type", "watch_hours", "last_login_days"]
    if not set(columnas_requeridas).issubset(df.columns):
        st.error("El CSV no tiene todas las columnas esperadas.")
        st.stop()
    col_filtro, col_seg = st.columns([1.3, 1])
    with col_filtro:
        planes_disponibles = sorted(df["subscription_type"].dropna().unique().tolist())
        planes = st.multiselect("Filtrar planes", planes_disponibles,
                               default=planes_disponibles)
    with col_seg:
        opciones = {
            "Tipo de suscripción": "subscription_type", "Dispositivo": "device",
            "Región": "region", "Género": "gender", "Género favorito": "favorite_genre"
        }
        opciones = {nombre: col for nombre, col in opciones.items() if col in df.columns}
        segmento = st.selectbox("Agrupar por", list(opciones))
    sub = df[df["subscription_type"].isin(planes)].copy()
    if sub.empty:
        st.info("Elige al menos un plan para ver resultados.")
        st.stop()
    n1, n2, n3 = st.columns(3)
    n1.metric("Clientes filtrados", f"{len(sub):,}")
    n2.metric("Cancelaron", f"{int((sub['churned'] == 1).sum()):,}")
    n3.metric("Abandono en el filtro", f"{sub['churned'].mean():.1%}")
    st.subheader("Tasa de abandono por segmento")
    seg = (sub.groupby(opciones[segmento])["churned"]
           .agg(clientes="size", tasa="mean").reset_index())
    seg["Abandono (%)"] = seg["tasa"] * 100
    seg = seg.sort_values("Abandono (%)")
    left, right = st.columns([1.6, 1])
    with left:
        st.plotly_chart(grafico_barras(seg, "Abandono (%)", opciones[segmento], True),
                        use_container_width=True)
    with right:
        st.dataframe(seg.rename(columns={opciones[segmento]: "Segmento", "clientes": "Clientes"})
                     [["Segmento", "Clientes", "Abandono (%)"]].round(1),
                     hide_index=True, use_container_width=True)
        st.caption("Tasa = clientes que cancelan / total de clientes del mismo segmento.")
    st.subheader("Comportamiento y correlaciones")
    left, right = st.columns(2)
    with left:
        fig = px.box(sub, x="churned", y="watch_hours", points=False,
                     color="churned", color_discrete_map={0: GRIS, 1: ROJO},
                     labels={"churned": "Abandono (0=no, 1=sí)",
                             "watch_hours": "Horas de visualización"})
        fig.update_layout(showlegend=False, height=330)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        num = [c for c in NUMERICAS if c in sub.columns]
        num += ["churned"]
        correlacion = sub[num].corr(method="spearman")
        fig = px.imshow(correlacion, text_auto=".2f", zmin=-1, zmax=1,
                        color_continuous_scale="RdBu_r", aspect="auto")
        fig.update_layout(height=360, margin=dict(l=5, r=5, t=8, b=20))
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Mapa de Spearman: asociación estadística; no indica causa-efecto.")


# ─── 3. Modelos y variables ──────────────────────────────────────────────────
elif pagina == "🤖 Comparar modelos":
    st.title("Evaluación de clasificadores")
    st.caption("Métricas guardadas en el notebook · conjunto de prueba: 1,000 clientes")
    metrica = st.selectbox("Comparar por métrica", ["F1", "Recall", "Precision", "Accuracy", "ROC-AUC"])
    tabla = RESULTADOS_TEST.sort_values(metrica, ascending=True)
    izquierda, derecha = st.columns([1.3, 1])
    with izquierda:
        st.plotly_chart(grafico_barras(tabla, metrica, "Modelo"), use_container_width=True)
    with derecha:
        st.dataframe(RESULTADOS_TEST.set_index("Modelo")
                     .sort_values("F1", ascending=False).style.format("{:.2%}"),
                     use_container_width=True)
        st.caption("F1 combina Precision y Recall. Una puntuación más alta es mejor.")

    st.subheader("¿Por qué estas variables?")
    st.write("El notebook prueba cuatro escenarios usando las variables seleccionadas. "
             "La mejor configuración por F1 promedio conserva todas.")
    c1, c2 = st.columns([1, 1.1])
    with c1:
        st.dataframe(ESCENARIOS_GB.style.format({"F1 CV": "{:.2%}"}),
                     hide_index=True, use_container_width=True)
        st.caption("F1 de Gradient Boosting en validación cruzada; la última fila "
                   "elimina las dos variables en revisión.")
    with c2:
        imp = IMPORTANCIAS.sort_values("Disminución del F1")
        st.plotly_chart(grafico_barras(imp, "Disminución del F1", "Variable"),
                        use_container_width=True)
        st.caption("Importancia por permutación (notebook). Cuanto más cae el F1 "
                   "al alterar una variable, mayor su aporte predictivo.")

    with st.expander("Ver variables finales y codificación"):
        st.write("**Variables originales:** `watch_hours`, `last_login_days`, "
                 "`avg_watch_time_per_day`, `number_of_profiles` y "
                 "`subscription_type`.")
        st.write("**Tras One-Hot Encoding:**")
        st.code("\n".join(COLUMNAS), language="text")
        st.caption("No se usan customer_id, age, gender, region, device, "
                   "monthly_fee, payment_method ni favorite_genre en estos modelos.")

    st.divider()
    st.subheader("Visualizaciones de evaluación")
    if df is None:
        st.info("Agrega el CSV para generar matrices de confusión, curvas ROC/PR "
                "y un árbol de decisión. Los resultados numéricos anteriores "
                "sí están disponibles desde el notebook.")
    else:
        try:
            evaluacion = entrenar_para_dashboard(df)
            modelo_nombre = st.selectbox("Modelo a inspeccionar", MODELOS, index=3)
            modelo = evaluacion["modelos"][modelo_nombre]
            x_test = evaluacion["x_test"]
            y_test = evaluacion["y_test"]
            pred = modelo.predict(x_test)
            prob = modelo.predict_proba(x_test)[:, 1]
            t1, t2, t3 = st.tabs(["Matriz de confusión", "Curvas ROC y PR", "Árbol"])
            with t1:
                cm = confusion_matrix(y_test, pred, labels=[0, 1])
                fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                x=["Permanece", "Abandona"],
                                y=["Permanece", "Abandona"])
                fig.update_layout(xaxis_title="Predicción", yaxis_title="Valor real",
                                  height=390, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("La diagonal muestra los aciertos; las otras casillas son errores.")
            with t2:
                fpr, tpr, _ = roc_curve(y_test, prob)
                prec, rec, _ = precision_recall_curve(y_test, prob)
                a, b = st.columns(2)
                with a:
                    roc = go.Figure()
                    roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=modelo_nombre,
                                             line=dict(color=ROJO, width=3)))
                    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                             line=dict(color=GRIS, dash="dash"), showlegend=False))
                    roc.update_layout(title="Curva ROC", xaxis_title="Tasa de falsos positivos",
                                      yaxis_title="Recall", height=345)
                    st.plotly_chart(roc, use_container_width=True)
                with b:
                    pr = go.Figure()
                    pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=modelo_nombre,
                                            line=dict(color=ROJO, width=3)))
                    pr.update_layout(title="Precision–Recall", xaxis_title="Recall",
                                     yaxis_title="Precision", height=345)
                    st.plotly_chart(pr, use_container_width=True)
            with t3:
                arbol = evaluacion["modelos"]["Árbol de Decisión"]
                fig, ax = plt.subplots(figsize=(16, 7))
                plot_tree(arbol, max_depth=3, filled=True, rounded=True, fontsize=8,
                          feature_names=COLUMNAS,
                          class_names=["Permanece", "Abandona"], ax=ax)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption("Se muestran solo los primeros niveles para facilitar la lectura.")
            st.caption("Visualizaciones recalculadas con el CSV y la misma partición "
                       "aleatoria que el notebook (semilla 42).")
        except Exception as exc:
            st.error(f"No se pudo preparar la evaluación: {exc}")


# ─── 4. Simulador ────────────────────────────────────────────────────────────
else:
    st.title("Simulador de riesgo de abandono")
    st.write("Introduce el comportamiento de un cliente para estimar la probabilidad "
             "de que pertenezca a la clase **churned = 1**.")
    st.caption("Se utiliza Gradient Boosting (modelo elegido en validación cruzada). "
               "Esta estimación es académica, no una decisión automatizada real.")
    modelo_pred = None
    columnas_modelo = COLUMNAS
    if bundle and "modelo" in bundle and "columnas" in bundle:
        modelo_pred = bundle["modelo"]
        columnas_modelo = list(bundle["columnas"])
        st.caption("Origen de la predicción: modelo .joblib exportado por el notebook.")
    elif df is not None:
        try:
            modelo_pred = entrenar_para_dashboard(df)["modelos"]["Gradient Boosting"]
            st.caption("Origen de la predicción: modelo reconstruido con el CSV "
                       "y los hiperparámetros del notebook.")
        except Exception as exc:
            st.warning(f"No se pudo entrenar el modelo: {exc}")
    if modelo_pred is None:
        st.warning("Se requiere modelo_churn_netflix.joblib o el CSV original "
                   "para habilitar las predicciones.")
        st.stop()

    with st.form("perfil_cliente"):
        a, b = st.columns(2)
        with a:
            horas = st.number_input("Horas totales de visualización", min_value=0.0,
                                    max_value=110.4, value=8.0, step=.5)
            dias = st.number_input("Días desde el último inicio de sesión",
                                   min_value=0, max_value=60, value=30)
        with b:
            perfiles = st.number_input("Número de perfiles", min_value=1,
                                       max_value=5, value=3)
            plan = st.selectbox("Tipo de suscripción", PLANES)
        enviado = st.form_submit_button("Calcular probabilidad de abandono", type="primary",
                                        use_container_width=True)
    umbral = st.slider("Umbral para clasificar como abandono", min_value=.10,
                      max_value=.90, value=.50, step=.05,
                      help="0.50 es el umbral predeterminado. Cambiarlo no recalibra el modelo.")
    if enviado:
        promedio = round(horas / (dias + 1), 2)
        entrada = {col: 0 for col in COLUMNAS}
        entrada.update({"watch_hours": horas, "last_login_days": dias,
                        "avg_watch_time_per_day": promedio,
                        "number_of_profiles": perfiles,
                        f"subscription_type_{plan}": 1})
        datos_entrada = pd.DataFrame([entrada])
        try:
            prob = float(modelo_pred.predict_proba(datos_entrada[columnas_modelo])[0, 1])
            st.session_state["ultima_prediccion"] = (prob, promedio)
        except Exception as exc:
            st.error(f"No se pudo estimar el abandono: {exc}")

    # Conserva el resultado al ajustar el umbral, sin repetir el entrenamiento.
    if "ultima_prediccion" in st.session_state:
        prob, promedio = st.session_state["ultima_prediccion"]
        st.subheader("Resultado de la estimación")
        a, b, c = st.columns(3)
        a.metric("Probabilidad estimada de abandono", f"{prob:.1%}")
        b.metric("Clasificación con el umbral", "Abandona" if prob >= umbral else "Permanece")
        c.metric("Promedio diario calculado", f"{promedio:.2f} h")
        st.progress(prob)
        if prob >= umbral:
            st.warning("Perfil priorizado para revisión en una campaña de retención.")
        else:
            st.success("Perfil por debajo del umbral de alerta seleccionado.")
        if promedio > 24:
            st.warning("El promedio diario supera 24 horas: revisa la coherencia de los datos.")
        st.caption("Promedio diario calculado como watch_hours / (last_login_days + 1), "
                   "según la relación observada en el dataset. Las probabilidades "
                   "pueden no estar calibradas; no implican causalidad.")

st.divider()
st.caption("Proyecto académico · Netflix Customer Churn · Fuentes: "
           "notebook netflix_churned.html y dataset del proyecto")
