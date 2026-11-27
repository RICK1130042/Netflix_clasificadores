"""Netflix | Churn Analytics — Dashboard Streamlit.

Basado en el notebook netflix_churned.ipynb. Las métricas de los modelos se leen
de los CSV que exporta la última celda del notebook, así el dashboard siempre
muestra los mismos resultados que el notebook:

    comparacion_validacion_cruzada.csv   F1, Recall, etc. en validación cruzada (5 folds)
    comparacion_modelos_churn.csv        métricas en el conjunto de prueba (20 %)
    importancia_variables_churn.csv      importancia por permutación del modelo ganador
    modelo_churn_netflix.joblib          modelo ganador ya entrenado
    etiquetas_clusters.csv              asignaciones por cliente y coordenadas PC1/PC2
    perfiles_clusters.csv               nombres de los seis clusters

Ejecuta: streamlit run app.py
"""
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))  # para encontrar ui.py se ejecute como se ejecute
import ui  # noqa: E402  componentes visuales compartidos
from palette import COLORES
import auth
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits


# ─── Configuración del proyecto ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
CSV_FILE = ROOT / "datos" / "netflix_validado.csv"
MODEL_FILE = ROOT / "modelo_churn_netflix.joblib"
CV_FILE = ROOT / "comparacion_validacion_cruzada.csv"
TEST_FILE = ROOT / "comparacion_modelos_churn.csv"
IMP_FILE = ROOT / "importancia_variables_churn.csv"
CLUSTER_FILE = ROOT / "etiquetas_clusters.csv"
PERFILES_FILE = ROOT / "perfiles_clusters.csv"

NUMERICAS = ["watch_hours", "last_login_days", "avg_watch_time_per_day", "number_of_profiles"]
PLANES = ["Basic", "Premium", "Standard"]
COLUMNAS = NUMERICAS + [f"subscription_type_{plan}" for plan in PLANES]
ESCENARIO_BASE = "Todas las seleccionadas"
K_SELECCIONADO = 6
VALORES_K = tuple(range(2, 11))
META_HIPOTESIS = 0.90          # F1 y Recall mínimos aprobados por el equipo (clase churned = 1)

ROJO = COLORES["rojo"]
GRIS = COLORES["gris"]
VERDE = COLORES["verde"]
AMBAR = COLORES["ambar"]
AZUL = COLORES["azul"]

# Nombres en español para las columnas del dataset
ETIQUETAS = {
    "watch_hours": "Horas vistas (h)",
    "last_login_days": "Días desde el último acceso",
    "avg_watch_time_per_day": "Horas promedio por día (h)",
    "number_of_profiles": "Número de perfiles",
    "monthly_fee": "Mensualidad ($)",
    "age": "Edad (años)",
    "churned": "Abandono (churned)",
}


st.set_page_config(page_title="Netflix · Churn Analytics", page_icon="🎬",
                   layout="wide", initial_sidebar_state="expanded")
ui.aplicar_estilos()
rol = auth.iniciar_sesion()


# ─── Carga de datos y resultados ────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def leer_csv(path: str, version: int) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def leer_modelo(path: str, version: int) -> dict:
    # Abrir SOLO el .joblib propio generado por el notebook.
    return joblib.load(path)


def validar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Revisa columnas, nulos y tipos antes de usar el dataset."""
    necesarias = NUMERICAS + ["subscription_type", "churned"]
    faltantes = [col for col in necesarias if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {', '.join(faltantes)}")
    if df[necesarias].isna().any().any():
        raise ValueError("El dataset contiene nulos en las variables del modelo.")
    if not df["churned"].isin([0, 1]).all():
        raise ValueError("churned debe contener únicamente 0 y 1.")
    df = df.copy()
    df["churned"] = df["churned"].astype(int)
    return df


def preparar_x(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Mismo One-Hot Encoding que el notebook, con el orden de columnas fijo."""
    x = pd.get_dummies(df[NUMERICAS + ["subscription_type"]],
                       columns=["subscription_type"], dtype=int
                       ).reindex(columns=COLUMNAS, fill_value=0)
    return x, df["churned"].astype(int)


def construir_modelos() -> dict:
    """Mismos hiperparámetros y preprocesamiento que en el notebook (celda de modelos)."""
    preprocesador = ColumnTransformer([("escalado", StandardScaler(), NUMERICAS)],
                                      remainder="passthrough")
    return {
        "Regresión Logística": Pipeline([
            ("preprocesamiento", preprocesador),
            ("clasificador", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=9, min_samples_leaf=5,
                                                random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=.06,
                                                        max_depth=3, random_state=42),
        "XGBoost": XGBClassifier(objective="binary:logistic", n_estimators=150, learning_rate=.05,
                                 max_depth=4, min_child_weight=3, subsample=.8, colsample_bytree=.8,
                                 reg_lambda=1.0, eval_metric="logloss", tree_method="hist",
                                 random_state=42, n_jobs=2),
    }


@st.cache_resource(show_spinner="Preparando los clasificadores con el CSV...")
def entrenar_para_dashboard(df: pd.DataFrame) -> dict:
    """Reconstruye los clasificadores sobre el split fijo del notebook (80/20, semilla 42).

    Se cachean: NO se reentrenan al mover un filtro. Solo se usan para las gráficas
    de evaluación (matriz de confusión, curvas y árbol); las métricas vienen del notebook.
    """
    x, y = preparar_x(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=42, stratify=y)
    modelos = construir_modelos()
    for modelo in modelos.values():
        modelo.fit(x_train, y_train)
    return {"modelos": modelos, "x_test": x_test, "y_test": y_test}


def cargar_resultados() -> dict | None:
    """Lee los CSV de resultados del notebook y define el modelo ganador por F1 en CV."""
    try:
        cv = leer_csv(str(CV_FILE), CV_FILE.stat().st_mtime_ns)
        test = leer_csv(str(TEST_FILE), TEST_FILE.stat().st_mtime_ns)
        imp = leer_csv(str(IMP_FILE), IMP_FILE.stat().st_mtime_ns)
    except FileNotFoundError as exc:
        st.error(f"Falta un archivo de resultados del notebook: {Path(exc.filename).name}")
        return None
    base = cv[cv["Escenario"] == ESCENARIO_BASE].sort_values("F1", ascending=False)
    ganador = base.iloc[0]["Modelo"]
    return {"cv": cv, "cv_base": base, "test": test, "imp": imp, "ganador": ganador,
            "test_ganador": test.set_index("Modelo").loc[ganador],
            "cv_ganador": base.set_index("Modelo").loc[ganador]}


def grafico_barras(tabla: pd.DataFrame, x: str, y: str, titulo: str, formato: str = "pct",
                   eje: str | None = None, resaltar: str | None = None):
    """Barras horizontales. formato: 'pct' (0.93 → 93.0 %), 'pct100' (ya en %) o 'dec'."""
    tabla = tabla.copy()
    if formato == "pct":
        tabla["_valor"] = tabla[x] * 100
        texto, sufijo = "%{x:.1f} %", " (%)"
    elif formato == "pct100":
        tabla["_valor"] = tabla[x]
        texto, sufijo = "%{x:.1f} %", " (%)"
    else:
        tabla["_valor"] = tabla[x]
        texto, sufijo = "%{x:.3f}", ""
    if tabla.empty:
        fig = go.Figure()
        fig.add_annotation(text="No hay valores suficientes para esta gráfica.", showarrow=False)
        return fig
    colores = [ROJO if (resaltar is None or v == resaltar) else GRIS for v in tabla[y]]
    fig = go.Figure(go.Bar(x=tabla["_valor"], y=tabla[y], orientation="h", marker_color=colores,
                           texttemplate=texto, textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: " + texto + "<extra></extra>"))
    minimo = min(0, tabla["_valor"].min())
    maximo = tabla["_valor"].max()
    margen = (maximo - minimo) * .18 or 1
    fig.update_layout(title=dict(text=titulo, x=0, font=dict(size=15)),
                      margin=dict(l=0, r=20, t=45, b=10), height=320,
                      yaxis_title=None, xaxis_title=(eje or x) + sufijo,
                      xaxis=dict(range=[minimo - (margen if minimo < 0 else 0), maximo + margen]),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def pct(valor: float, decimales: int = 2) -> str:
    return f"{valor * 100:.{decimales}f} %"

# ─── K-Means: segmentación y evaluación predictiva ──────────────────────────
@st.cache_data(show_spinner="Calculando K-Means y curvas ROC para K = 2 a 10...")
def evaluar_kmeans(df: pd.DataFrame) -> dict:
    """Replica los dos bloques de Colab usando las variables del dashboard.

    Los perfiles de los seis grupos son descriptivos (dataset completo).
    Para ROC, tanto el escalado como K-Means se ajustan solo con entrenamiento.
    """
    x, y = preparar_x(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=.20, random_state=42, stratify=y
    )
    if len(x_train) < max(VALORES_K) or len(x) <= max(VALORES_K):
        raise ValueError("El dataset es demasiado pequeño para evaluar K de 2 a 10.")

    # Estandarizar las cuatro variables para los perfiles descriptivos de K = 6.
    x_cluster = StandardScaler().fit_transform(x[NUMERICAS])

    # Bloque 2: escalado independiente para la evaluación predictiva 80/20.
    scaler_cluster = StandardScaler()
    train_scaled = scaler_cluster.fit_transform(x_train[NUMERICAS])
    test_scaled = scaler_cluster.transform(x_test[NUMERICAS])

    resultados = []
    resultados_codo = []
    curvas = {}
    # Limitar hilos evita saturar el servidor cuando se prueban varios K.
    with threadpool_limits(limits=1):
        # Método del codo: replica el primer bloque de Colab con el dataset completo.
        # La inercia se utiliza únicamente para esta gráfica.
        for k in VALORES_K:
            descriptivo = KMeans(n_clusters=k, random_state=42, n_init=10)
            etiquetas = descriptivo.fit_predict(x_cluster)
            resultados_codo.append({"K": k, "Inercia": float(descriptivo.inertia_)})
            if k == K_SELECCIONADO:
                etiquetas_elegidas = etiquetas
        for k in VALORES_K:
            # Distancias a centroides obtenidos SOLO del conjunto de train.
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(train_scaled)
            nombres = [f"dist_cluster_{i}" for i in range(k)]
            train_dist = pd.DataFrame(kmeans.transform(train_scaled),
                                      columns=nombres, index=x_train.index)
            test_dist = pd.DataFrame(kmeans.transform(test_scaled),
                                     columns=nombres, index=x_test.index)
            x_train_k = pd.concat([x_train, train_dist], axis=1)
            x_test_k = pd.concat([x_test, test_dist], axis=1)
            modelo = Pipeline([
                ("scaler", StandardScaler()),
                ("logistic", LogisticRegression(max_iter=2000, random_state=42)),
            ])
            modelo.fit(x_train_k, y_train)
            prob = modelo.predict_proba(x_test_k)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc = float(roc_auc_score(y_test, prob))
            curvas[k] = {"fpr": fpr, "tpr": tpr, "auc": auc}
            resultados.append({"K": k, "ROC-AUC": auc})

    perfiles = df[NUMERICAS + ["churned"]].copy()
    perfiles["Cluster"] = etiquetas_elegidas + 1  # Etiquetas visibles: 1 a 6.
    segmentos = perfiles.groupby("Cluster").agg(
        Clientes=("churned", "size"),
        Abandono=("churned", "mean"),
        **{col: (col, "mean") for col in NUMERICAS},
    ).reindex(range(1, K_SELECCIONADO + 1)).reset_index()
    segmentos["Clientes"] = segmentos["Clientes"].fillna(0).astype(int)
    return {"resultados": pd.DataFrame(resultados), "curvas": curvas,
            "segmentos": segmentos, "n_test": len(y_test),
            "codo": pd.DataFrame(resultados_codo)}


def grafico_kmeans(tabla: pd.DataFrame, metrica: str, titulo: str):
    elegido = tabla.loc[tabla["K"] == K_SELECCIONADO].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tabla["K"], y=tabla[metrica],
                             mode="lines+markers", name=metrica,
                             line=dict(color=GRIS, width=2), marker=dict(size=8)))
    fig.add_vline(x=K_SELECCIONADO, line_dash="dash", line_color=ROJO)
    fig.add_trace(go.Scatter(x=[K_SELECCIONADO], y=[elegido[metrica]],
                             mode="markers", name="K = 6 seleccionado",
                             marker=dict(color=ROJO, size=15, symbol="diamond")))
    fig.update_layout(title=titulo, xaxis_title="Número de clusters (K)",
                      yaxis_title=metrica, height=350,
                      margin=dict(l=15, r=15, t=65, b=15),
                      legend=dict(orientation="h", y=-.25))
    fig.update_xaxes(tickmode="array", tickvals=list(VALORES_K))
    return fig


def mostrar_kmeans(df: pd.DataFrame):
    st.divider()
    st.subheader("K-Means | Segmentación de clientes")
    st.write("Evaluación de K = 2 a 10 para segmentar suscriptores y medir "
             "el aporte predictivo de las distancias a los centroides.")
    st.caption("Variables: watch_hours, last_login_days, avg_watch_time_per_day "
               "y number_of_profiles. Estandarización con StandardScaler.")
    if df is None:
        st.info("Carga netflix_customer_churn.csv para calcular las métricas y gráficas de K-Means.")
        return
    try:
        resultado = evaluar_kmeans(df)
    except Exception as exc:
        st.error(f"No se pudo evaluar K-Means: {exc}")
        return
    tabla = resultado["resultados"]
    curvas = resultado["curvas"]
    elegido = tabla.loc[tabla["K"] == K_SELECCIONADO].iloc[0]
    mejor_auc = tabla.loc[tabla["ROC-AUC"].idxmax()]
    st.success("Configuración seleccionada por el proyecto: K = 6 clusters.")
    a, b = st.columns(2)
    a.metric("Clusters seleccionados", str(K_SELECCIONADO))
    b.metric("ROC-AUC · K = 6", f'{elegido["ROC-AUC"]:.4f}')

    t_codo, t1, t2 = st.tabs(["Método del codo", "Curvas ROC y AUC", "Resultados y 6 clusters"])
    with t_codo:
        ui.grafica('¿A partir de qué número de clústeres disminuye la mejora de la compactación de grupos?', 
            grafico_kmeans(resultado["codo"], "Inercia", "Método del codo — K-Means"),
            use_container_width=True, key="km_codo"
        )
        st.caption("La gráfica muestra cómo disminuye la dispersión dentro de los grupos "
                   "al aumentar K. Se busca el punto donde la mejora empieza a ser menor. "
                   "K = 6 se destaca como la configuración seleccionada por el equipo; "
                   "la marca no implica que el codo se haya detectado automáticamente.")
    with t1:
        st.write("**K-Means + Regresión Logística**: se agregan las distancias a los "
                 "K centroides a las variables originales para estimar la probabilidad de churn.")
        roc = go.Figure()
        k_otro = st.selectbox("Comparar K = 6 con", [k for k in VALORES_K if k != K_SELECCIONADO], key="km_comparador")
        curva = curvas[k_otro]
        roc.add_trace(go.Scatter(x=curva["fpr"], y=curva["tpr"], mode="lines",
                                 name=f'K = {k_otro} | AUC = {curva["auc"]:.4f}',
                                 line=dict(color=AZUL, width=3, dash="dot")))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 name="Aleatorio | AUC = 0.50",
                                 line=dict(color=GRIS, dash="dash")))
        curva = curvas[K_SELECCIONADO]
        roc.add_trace(go.Scatter(x=curva["fpr"], y=curva["tpr"], mode="lines",
                                 name=f'K = 6 seleccionado | AUC = {curva["auc"]:.4f}',
                                 line=dict(color=ROJO, width=4)))
        roc.update_layout(title="Curvas ROC según número de clusters",
                          xaxis_title="Tasa de falsos positivos (FPR)",
                          yaxis_title="Tasa de verdaderos positivos (Recall)",
                          height=490, legend=dict(orientation="h", y=-.2))
        roc.update_xaxes(range=[0, 1])
        roc.update_yaxes(range=[0, 1.01])
        ui.grafica('¿Cómo cambia la discriminación del abandono al usar K = 6 frente a otro K?', roc, use_container_width=True, key="km_roc")
        ui.grafica('¿Qué cantidad de clústeres merece validación adicional por su AUC exploratorio?', grafico_kmeans(tabla, "ROC-AUC", "ROC-AUC según cantidad de clusters"),
                        use_container_width=True, key="km_auc")
        st.caption(f'Mayor AUC observado: K = {int(mejor_auc["K"])} '
                   f'({mejor_auc["ROC-AUC"]:.4f}). Partición 80/20 estratificada '
                   f'(semilla 42); prueba: {resultado["n_test"]:,} clientes.')
        st.info("K-Means es no supervisado y no tiene ROC-AUC propio. La métrica "
                "corresponde a la regresión logística con las distancias a centroides. "
                "La comparación de K sobre prueba es exploratoria; para seleccionar K "
                "por AUC se requiere validación independiente o cruzada en entrenamiento.")
    with t2:
        detalle = tabla.copy()
        detalle["Selección"] = np.where(detalle["K"] == K_SELECCIONADO, "Seleccionado", "")
        st.dataframe(detalle.style.format({"ROC-AUC": "{:.4f}"})
                     .apply(lambda fila: [f"font-weight: bold; color: {ROJO}"
                                          if fila["K"] == K_SELECCIONADO else ""] * len(fila), axis=1),
                     hide_index=True, use_container_width=True)
        st.markdown("**Perfil de los seis clusters seleccionados**")
        segmentos = resultado["segmentos"]
        fig = px.bar(segmentos, x="Cluster", y="Clientes", text="Clientes",
                     color_discrete_sequence=[AZUL], hover_data={"Abandono": ":.1%"})
        fig.update_xaxes(tickmode="array", tickvals=list(range(1, K_SELECCIONADO + 1)))
        fig.update_layout(height=350)
        ui.grafica('¿Qué tamaño y tasa de abandono tiene cada grupo para orientar la capacidad de retención?', fig, use_container_width=True, key="km_segmentos")
        st.dataframe(segmentos.style.format({"Abandono": "{:.1%}",
                                             **{col: "{:.2f}" for col in NUMERICAS}}),
                     hide_index=True, use_container_width=True)
        st.caption("Segmentación descriptiva del dataset completo. La tasa de abandono "
                   "se calcula después de formar los grupos; churned no se usa para agrupar. "
                   "Los números de cluster son etiquetas, no un orden de riesgo.")
    st.caption("Se conserva K = 6 como decisión del proyecto; los máximos observados "
               "pueden corresponder a otros valores de K. El modelo campeón original "
               "y el simulador permanecen sin cambios.")


# ─── Dataset y recursos disponibles ─────────────────────────────────────────
df = None
fecha_datos = None
if CSV_FILE.exists():
    try:
        df = validar_dataset(leer_csv(str(CSV_FILE), CSV_FILE.stat().st_mtime_ns))
        fecha_datos = datetime.fromtimestamp(CSV_FILE.stat().st_mtime)
    except Exception as exc:
        st.error(f"No fue posible leer el CSV: {exc}")
else:
    st.sidebar.caption("No se encontró datos/netflix_validado.csv en el repositorio.")
    archivo = st.sidebar.file_uploader("Cargar dataset CSV", type="csv") if rol == "analista" else None
    if archivo is not None:
        try:
            df = validar_dataset(pd.read_csv(archivo))
            fecha_datos = datetime.now()
        except Exception as exc:
            st.sidebar.error(f"CSV inválido: {exc}")

# ─── Segmentación validada por Itzel ───
perfiles = None
df_segmentado = df
clusters_disponibles = False
if df is not None:
    try:
        etiquetas = leer_csv(str(CLUSTER_FILE), CLUSTER_FILE.stat().st_mtime_ns)
        perfiles = leer_csv(str(PERFILES_FILE), PERFILES_FILE.stat().st_mtime_ns)
        if not etiquetas["cluster"].isin(range(1, 7)).all():
            raise ValueError("Etiquetas fuera del rango 1–6.")
        if set(perfiles["cluster"]) != set(range(1, 7)) or not perfiles["cluster"].is_unique:
            raise ValueError("Perfiles de clusters inválidos.")
        if perfiles["nombre"].isna().any():
            raise ValueError("Hay perfiles sin nombre.")
        if etiquetas["customer_id"].isna().any() or df["customer_id"].isna().any():
            raise ValueError("Identificadores faltantes.")
        unido = df.drop(columns=["cluster", "PC1", "PC2"], errors="ignore").merge(
            etiquetas[["customer_id", "cluster", "PC1", "PC2"]],
            on="customer_id", how="left", validate="one_to_one"
        )
        if unido["cluster"].isna().any():
            raise ValueError("Hay clientes sin asignación de cluster.")
        df_segmentado = unido
        clusters_disponibles = True
    except Exception as exc:
        st.sidebar.warning(f"Segmentación no disponible: {exc}")

res = cargar_resultados()

bundle = None
if MODEL_FILE.exists():
    try:
        bundle = leer_modelo(str(MODEL_FILE), MODEL_FILE.stat().st_mtime_ns)
    except Exception as exc:
        st.sidebar.warning(f"No se pudo cargar el modelo guardado: {exc}")

ui.marca("CHURN ANALYTICS · MODELOS DE CLASIFICACIÓN")
pagina = st.sidebar.radio("Navegación", auth.PAGINAS[rol], label_visibility="collapsed", key="navegacion")
auth.exigir_pagina(rol, pagina)
st.sidebar.divider()
st.sidebar.caption("Fuente de entrega: datos/netflix_validado.csv · tratamiento en contrato-datos.md")
st.sidebar.caption(f"Dataset: {'disponible' if df is not None else 'sin cargar'}")
if fecha_datos:
    st.sidebar.caption(f"Datos actualizados al: {fecha_datos:%d/%m/%Y}")
st.sidebar.caption(f"Modelo exportado: {bundle['nombre_modelo'] if bundle else 'sin cargar'}")
st.sidebar.caption("Métricas exportadas: cohorte histórica de 5,000 filas · prueba 80/20 · CV 5 folds")




# ─── Selector global de segmentos ───
df_filtrado = df_segmentado
elegido = 0
if pagina in ["🏠 Resumen ejecutivo", "📊 Explorar clientes"] and clusters_disponibles:
    nombres = perfiles.set_index("cluster")["nombre"].to_dict()
    ui.seccion("Segmentación de clientes")
    elegido = st.segmented_control(
        "Selecciona un segmento", options=list(range(7)),
        format_func=lambda x: "Todos" if x == 0 else f"Clúster {x}",
        default=0, key="cluster_global"
    )
    if elegido is None:
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
    st.caption("Segmentos según etiquetas_clusters.csv y perfiles_clusters.csv.")
    st.divider()

# ─── 1. Resumen ─────────────────────────────────────────────────────────────
if pagina == "🏠 Resumen ejecutivo":
    ui.encabezado_pagina("Panorama del abandono",
                         "Netflix necesita identificar qué comportamientos se asocian con la cancelación "
                         "para detectar segmentos vulnerables y orientar acciones de retención.",
                         etiqueta="Resumen ejecutivo")
    if df is None or res is None:
        st.warning("Faltan el dataset o los resultados del notebook para mostrar el panorama.")
        st.stop()

    ganador, t_gan, cv_gan = res["ganador"], res["test_ganador"], res["cv_ganador"]
    meta_ok = "ok" if (t_gan["F1"] >= META_HIPOTESIS and t_gan["Recall"] >= META_HIPOTESIS) else "alerta"
    ui.fila_kpis([
        ("Clientes analizados", f"{len(df_filtrado):,}",
         f"Clúster {elegido}" if elegido not in (None, 0) else "dataset completo de Kaggle"),
        ("Tasa de abandono", pct(df_filtrado["churned"].mean(), 1), f"{int(df_filtrado['churned'].sum()):,} clientes cancelaron", "alerta"),
        (f"F1 de {ganador}", pct(t_gan["F1"]), f"prueba · meta ≥ {META_HIPOTESIS:.0%}", meta_ok),
        (f"Recall de {ganador}", pct(t_gan["Recall"]), f"prueba · meta ≥ {META_HIPOTESIS:.0%}", meta_ok),
    ])

    if clusters_disponibles:
        st.caption("F1 y Recall corresponden a la evaluación global, no al clúster.")
        ui.seccion("Comportamiento del segmento seleccionado")
        p, q, r = st.columns(3)
        p.metric("Horas promedio", f"{df_filtrado['watch_hours'].mean():.1f} h")
        q.metric("Días sin acceso", f"{df_filtrado['last_login_days'].mean():.1f}")
        r.metric("Perfiles promedio", f"{df_filtrado['number_of_profiles'].mean():.1f}")
        if elegido not in (None, 0):
            st.caption("Los patrones, hallazgos y métricas de modelos que siguen corresponden al dataset completo.")

    # ¿Qué modelo se seleccionó?
    ui.seccion("¿Qué modelo se seleccionó?")
    izq, der = st.columns([1.4, 1])
    with izq:
        segundo = res["cv_base"].iloc[1]
        mejor_test = res["test"].sort_values("F1", ascending=False).iloc[0]
        st.write(f"**{ganador}** obtuvo el F1 promedio más alto en validación cruzada "
                 f"(**{pct(cv_gan['F1'])}**, contra {pct(segundo['F1'])} de {segundo['Modelo']}). "
                 f"La selección se hizo solo con el conjunto de entrenamiento, para no usar los datos "
                 f"de prueba al elegir.")
        st.write(f"En el conjunto de prueba (1,000 clientes que el modelo no vio) alcanzó "
                 f"**F1 de {pct(t_gan['F1'])}** y **Recall de {pct(t_gan['Recall'])}** en la clase de abandono.")
        if mejor_test["Modelo"] != ganador:
            st.write(f"En prueba, **{mejor_test['Modelo']}** quedó ligeramente arriba "
                     f"(F1 de {pct(mejor_test['F1'])}). Se mantiene {ganador} porque cambiar de modelo "
                     f"después de ver la prueba sería elegir con esos datos.")
        cumple = t_gan["F1"] >= META_HIPOTESIS and t_gan["Recall"] >= META_HIPOTESIS
        mensaje = (f"F1 y Recall ≥ {META_HIPOTESIS:.0%} en la clase churned = 1 "
                   f"(F1 {pct(t_gan['F1'])} · Recall {pct(t_gan['Recall'])}).")
        if cumple:
            st.success("Hipótesis de desempeño cumplida: " + mensaje)
        else:
            st.error("Hipótesis de desempeño no cumplida: se esperaba " + mensaje)
        st.caption("En SAS Model Studio el campeón fue Forest porque allí el criterio fue el KS. "
                   "Aquí el criterio aprobado es el F1 en validación cruzada.")
    with der:
        graf = res["cv_base"].sort_values("F1")
        ui.grafica('¿Qué modelo se debe seleccionar por F1 en validación cruzada?', grafico_barras(graf, "F1", "Modelo", "F1 promedio en validación cruzada",
                                       eje="F1", resaltar=ganador), use_container_width=True)
        st.caption("5 particiones estratificadas sobre el 80 % de entrenamiento.")

    # Patrones del análisis exploratorio
    ui.seccion("Patrones principales del análisis exploratorio")
    med = df.groupby("churned")[["watch_hours", "last_login_days", "avg_watch_time_per_day"]].median()
    ui.fila_kpis([
        ("Horas vistas · permanece", f"{med.loc[0, 'watch_hours']:.2f} h", "mediana, churned = 0"),
        ("Horas vistas · cancela", f"{med.loc[1, 'watch_hours']:.2f} h", "mediana, churned = 1", "alerta"),
        ("Días sin acceso · permanece", f"{med.loc[0, 'last_login_days']:.0f} días", "mediana, churned = 0"),
        ("Días sin acceso · cancela", f"{med.loc[1, 'last_login_days']:.0f} días", "mediana, churned = 1", "alerta"),
    ])

    t_izq, t_der = st.columns([1, 1.3])
    with t_izq:
        comp = med.T.rename(columns={0: "Permanece", 1: "Cancela"})
        comp.index = [ETIQUETAS[i] for i in comp.index]
        comp["Diferencia"] = comp["Cancela"] - comp["Permanece"]
        st.dataframe(comp.style.format("{:.2f}"), width="stretch")
        st.caption("Medianas por grupo. Se usa la mediana porque las horas tienen valores atípicos altos.")
    with t_der:
        razon = med.loc[0, "watch_hours"] / med.loc[1, "watch_hours"]
        st.write(f"Quien permanece ve una mediana de **{razon:.1f} veces más horas** que quien cancela "
                 f"({med.loc[0, 'watch_hours']:.2f} h contra {med.loc[1, 'watch_hours']:.2f} h) y lleva "
                 f"**{med.loc[1, 'last_login_days'] - med.loc[0, 'last_login_days']:.0f} días menos** sin entrar "
                 f"({med.loc[0, 'last_login_days']:.0f} contra {med.loc[1, 'last_login_days']:.0f}).")
        st.write("Los dos grupos se separan sobre todo por **actividad**: cuánto contenido ven y cada cuánto "
                 "entran. Para el negocio, eso significa que el riesgo se puede detectar con datos de uso "
                 "que ya se registran, antes de que el cliente cancele.")
        ui.nota("<b>Asociación, no causalidad:</b> estos datos no demuestran que ver menos horas cause la cancelación.", "aviso")

    # Hallazgos
    ui.seccion("Hallazgos y oportunidades")
    plan = df.groupby("subscription_type")["churned"].mean().sort_values(ascending=False)
    zona = (df["watch_hours"] < 5) & (df["last_login_days"] > 30)
    imp_top = res["imp"].sort_values("Importancia", ascending=False)["Variable"].head(3).tolist()
    h1, h2, h3 = st.columns(3)
    ui.tarjeta_lista("📈 Hallazgos estadísticos", [
        f"El plan <b>{plan.index[0]}</b> tiene la mayor tasa de abandono ({plan.iloc[0]:.1%}), "
        f"contra {plan.iloc[-1]:.1%} de {plan.index[-1]}.",
        f"Los <b>{zona.sum():,}</b> clientes con menos de 5 h vistas y más de 30 días sin entrar "
        f"cancelaron en un {df.loc[zona, 'churned'].mean():.0%}.",
        "Mayor asociación con abandono (Spearman): horas promedio por día, horas vistas y días sin acceso.",
    ], h1)
    ui.tarjeta_lista("🤖 Predicciones", [
        f"{ganador} identifica a {t_gan['Recall']:.0%} de los clientes que cancelan en prueba.",
        f"De cada 100 clientes que marca como riesgo, {t_gan['Precision'] * 100:.0f} sí cancelan.",
        "Variables con más aporte al modelo: " + ", ".join(f"<code>{v}</code>" for v in imp_top) + ".",
    ], h2)
    ui.tarjeta_lista("💡 Recomendaciones (por validar)", [
        "Priorizar el análisis de clientes Basic con baja actividad.",
        "Vigilar a quien supera 30 días sin acceso, antes del siguiente cobro.",
        "Probar recomendaciones de contenido en clientes con pocas horas vistas.",
        "Medir el efecto con un grupo de control: no hay evidencia aún de que reduzcan el abandono.",
    ], h3)


# ─── 2. Exploración de clientes ──────────────────────────────────────────────
elif pagina == "📊 Explorar clientes":
    ui.encabezado_pagina("Explorar el comportamiento",
                         "Filtra por plan y agrupa por segmento; los indicadores se recalculan al instante.",
                         etiqueta="Explorar clientes")
    if df is None:
        st.warning("Para mostrar gráficas interactivas carga netflix_customer_churn.csv.")
        st.stop()

    planes_disponibles = sorted(df["subscription_type"].dropna().unique().tolist())
    opciones = {"Tipo de suscripción": "subscription_type", "Dispositivo": "device", "Región": "region",
                "Género": "gender", "Género favorito": "favorite_genre", "Método de pago": "payment_method",
                "Número de perfiles": "number_of_profiles"}
    opciones = {nombre: col for nombre, col in opciones.items() if col in df.columns}

    def restablecer_filtros():
        st.session_state["f_planes"] = planes_disponibles
        st.session_state["f_segmento"] = "Tipo de suscripción"

    st.session_state.setdefault("f_planes", planes_disponibles)
    st.session_state.setdefault("f_segmento", "Tipo de suscripción")

    col_filtro, col_seg, col_btn = st.columns([1.3, 1, .45])
    with col_filtro:
        planes = st.multiselect("Filtrar planes", planes_disponibles, key="f_planes")
    with col_seg:
        segmento = st.selectbox("Agrupar por", list(opciones), key="f_segmento")
    with col_btn:
        st.write("")
        st.button("Restablecer filtros", on_click=restablecer_filtros, width="stretch")

    sub = df_filtrado[df_filtrado["subscription_type"].isin(planes)].copy()
    filtro_cluster = "Todos" if elegido == 0 else f"Clúster {elegido}"
    st.caption(f"Planes activos: {', '.join(planes) or 'Ninguno'} · Segmento: {filtro_cluster} · Agrupación: {segmento}")
    st.caption(f"Clientes filtrados: {len(sub):,} de {len(df):,}")
    if sub.empty:
        st.info("No hay clientes con este filtro. Elige al menos un plan con registros.")
        st.stop()

    ui.fila_kpis([
        ("Clientes filtrados", f"{len(sub):,}", f"de {len(df):,} en total"),
        ("Cancelaron", f"{int(sub['churned'].sum()):,}", "churned = 1"),
        ("Abandono en el filtro", pct(sub["churned"].mean(), 1),
         f"general: {pct(df['churned'].mean(), 1)}", "alerta"),
    ])

    ui.seccion("Tasa de abandono por segmento")
    col = opciones[segmento]
    seg = (sub.groupby(col)["churned"].agg(clientes="size", cancelan="sum", tasa="mean").reset_index())
    if seg.empty:
        st.info("La agrupación no contiene categorías válidas.")
        st.stop()
    seg[col] = seg[col].astype(str)
    seg["Abandono (%)"] = seg["tasa"] * 100
    seg = seg.sort_values("Abandono (%)", ascending=False)        # mayor a menor
    left, right = st.columns([1.5, 1])
    with left:
        ui.grafica('¿Qué segmento presenta mayor tasa de abandono y merece atención prioritaria?', grafico_barras(seg.iloc[::-1], "Abandono (%)", col,
                                       f"Tasa de abandono por {segmento.lower()}", formato="pct100",
                                       eje="Abandono", resaltar=seg.iloc[0][col]), use_container_width=True)
    with right:
        tabla_seg = seg.rename(columns={col: "Segmento", "clientes": "Clientes", "cancelan": "Cancelan"})
        st.dataframe(tabla_seg[["Segmento", "Clientes", "Cancelan", "Abandono (%)"]]
                     .style.format({"Clientes": "{:,}", "Cancelan": "{:,}", "Abandono (%)": "{:.1f} %"}),
                     hide_index=True, column_order=["Segmento", "Clientes", "Cancelan", "Abandono (%)"], width="stretch")
        st.caption("CALCULADO · fuente: CSV activo del dashboard; agrupación por segmento. "
                   "Tasa = clientes que cancelan / total de clientes del mismo segmento. "
                   "Ordenado de mayor a menor abandono.")
        brecha = seg["Abandono (%)"].max() - seg["Abandono (%)"].min()
        st.write(f"**{seg.iloc[0][col]}** es el segmento con más abandono ({seg.iloc[0]['Abandono (%)']:.1f} %). "
                 f"La diferencia entre el segmento más alto y el más bajo es de {brecha:.1f} puntos"
                 + ("; es pequeña, así que esta variable distingue poco el riesgo." if brecha < 5 else "."))
        if rol == "analista":
            st.download_button("Descargar clientes filtrados (CSV)",
                               sub.to_csv(index=False).encode("utf-8"),
                               file_name="clientes_filtrados.csv", mime="text/csv", width="stretch")
        else:
            st.caption("La exportación de registros individuales está reservada al analista.")

    ui.seccion("Comportamiento y correlaciones")
    left, right = st.columns(2)
    with left:
        caja = sub.assign(Estado=sub["churned"].map({0: "0 · Permanece", 1: "1 · Cancela"}))
        fig = px.box(caja, x="Estado", y="watch_hours", points=False, color="Estado",
                     color_discrete_map={"0 · Permanece": AZUL, "1 · Cancela": ROJO},
                     category_orders={"Estado": ["0 · Permanece", "1 · Cancela"]},
                     labels={"Estado": "Estado de abandono (0 = permanece, 1 = cancela)",
                             "watch_hours": "Horas de visualización (h)"})
        fig.update_layout(title=dict(text="Horas de visualización según abandono", x=0, font=dict(size=15)),
                          showlegend=False, height=380, margin=dict(t=45))
        ui.grafica('¿Qué diferencias de horas vistas existen entre quienes permanecen y quienes cancelan?', fig, use_container_width=True)
        atipicos = {}
        for estado, grupo in sub.groupby("churned")["watch_hours"]:
            q1, q3 = grupo.quantile([.25, .75])
            atipicos[estado] = int((grupo > q3 + 1.5 * (q3 - q1)).sum())
        st.caption(f"Atípicos (regla 1.5 × RIC): {atipicos.get(0, 0)} en quienes permanecen "
                   f"y {atipicos.get(1, 0)} en quienes cancelan. No se muestran puntos individuales.")
    with right:
        num = [c for c in ["watch_hours", "last_login_days", "avg_watch_time_per_day",
                           "number_of_profiles", "monthly_fee", "age"] if c in sub.columns] + ["churned"]
        correlacion = sub[num].corr(method="spearman")
        cortos = {"watch_hours": "Horas vistas", "last_login_days": "Días sin acceso",
                  "avg_watch_time_per_day": "Horas/día", "number_of_profiles": "Perfiles",
                  "monthly_fee": "Mensualidad", "age": "Edad", "churned": "Abandono"}
        nombres = [cortos.get(c, c) for c in num]
        rel_grafica = correlacion["churned"].drop("churned").dropna().sort_values(key=abs)
        tabla_corr = pd.DataFrame({"Variable": [cortos.get(c, c) for c in rel_grafica.index], "rho": rel_grafica.values})
        fig = grafico_barras(tabla_corr, "rho", "Variable", "Correlación de Spearman con abandono", formato="dec", eje="ρ")
        ui.grafica('¿Qué variables se asocian más con el abandono y merecen seguimiento?', fig, use_container_width=True)

    rel = correlacion["churned"].drop("churned").sort_values(key=abs, ascending=False)
    fuertes = rel[rel.abs() >= .3]
    debiles = rel[rel.abs() < .1]
    lineas = [f"**{ETIQUETAS.get(v, v)}** ({c:+.2f}): " +
              ("a más valor, más abandono." if c > 0 else "a más valor, menos abandono.")
              for v, c in fuertes.items()]
    st.markdown("**Relaciones más fuertes con el abandono:**\n\n- " + "\n- ".join(lineas) if lineas
                else "No hay variables con correlación moderada o fuerte en este filtro.")
    if len(debiles):
        st.write("Casi sin relación (|ρ| < 0.10): " + ", ".join(ETIQUETAS.get(v, v) for v in debiles.index) + ".")
    if elegido in (None, 0):
        st.write("Las variables de actividad son las más útiles para detectar riesgo. Horas vistas y horas "
                 "promedio por día están muy relacionadas entre sí, así que aportan información parecida.")
    ui.nota("<b>Correlación no significa causalidad:</b> estas relaciones describen los datos, no prueban causas.", "aviso")


# ─── 3. Modelos y variables ──────────────────────────────────────────────────
elif pagina == "🤖 Comparar modelos":
    ui.encabezado_pagina("Evaluación de clasificadores",
                         "Cómo se comportan los modelos con clientes que no vieron durante el entrenamiento.",
                         etiqueta="Comparar modelos")
    if res is None:
        st.stop()
    st.caption(f"Métricas exportadas por el notebook · conjunto de prueba: 1,000 clientes · "
               f"clase positiva: churned = 1 · modelo elegido: {res['ganador']}")
    metrica = st.selectbox("Comparar por métrica", ["F1", "Recall", "Precision", "Accuracy", "ROC-AUC"])
    tabla = res["test"].sort_values(metrica, ascending=True)
    izquierda, derecha = st.columns([1.3, 1])
    with izquierda:
        ui.grafica('¿Cómo se comparan los modelos en prueba según la métrica seleccionada?', grafico_barras(tabla, metrica, "Modelo", f"{metrica} en el conjunto de prueba",
                                       eje=metrica, resaltar=res["ganador"]), use_container_width=True)
    with derecha:
        st.dataframe(res["test"].set_index("Modelo").sort_values("F1", ascending=False)
                     .style.format("{:.2%}"), width="stretch")
        st.caption("Precision: de los clientes marcados como abandono, cuántos sí cancelan. "
                   "Recall: de los que cancelan, cuántos detecta el modelo. F1 combina ambas.")

    mostrar_kmeans(df)
    if clusters_disponibles:
        st.caption("La evaluación K-Means anterior recalcula los clusters con el dataset completo. "
                   "El selector de Resumen y Explorar usa las etiquetas exportadas; "
                   "la numeración de ambos análisis no tiene por qué coincidir.")

    ui.seccion("¿Por qué estas variables?")
    st.write(f"El notebook prueba cuatro escenarios de variables con validación cruzada. "
             f"La mejor configuración por F1 promedio conserva todas.")
    c1, c2 = st.columns([1, 1.1])
    with c1:
        esc = (res["cv"][res["cv"]["Modelo"] == res["ganador"]][["Escenario", "F1", "Recall"]]
               .sort_values("F1", ascending=False))
        st.dataframe(esc.style.format({"F1": "{:.2%}", "Recall": "{:.2%}"}), hide_index=True, width="stretch")
        st.caption(f"F1 y Recall de {res['ganador']} en validación cruzada para cada escenario.")
    with c2:
        imp = res["imp"].sort_values("Importancia")
        ui.grafica('¿Qué variables aportan más al desempeño del modelo y conviene vigilar?', grafico_barras(imp, "Importancia", "Variable",
                                       f"Importancia por permutación · {res['ganador']}", formato="dec",
                                       eje="Disminución del F1", resaltar=imp.iloc[-1]["Variable"]),
                        use_container_width=True)
        st.caption("Cuanto más cae el F1 al alterar una variable, mayor su aporte predictivo. "
                   "Valores cercanos a 0 indican que la variable casi no aporta.")

    with st.expander("Ver variables finales y codificación"):
        st.write("**Variables originales:** `watch_hours`, `last_login_days`, `avg_watch_time_per_day`, "
                 "`number_of_profiles` y `subscription_type`.")
        st.write("**Tras One-Hot Encoding:**")
        st.code("\n".join(COLUMNAS), language="text")
        st.caption("No se usan customer_id, age, gender, region, device, monthly_fee, payment_method "
                   "ni favorite_genre en estos modelos.")

    ui.seccion("Visualizaciones de evaluación")
    if df is None:
        st.info("Agrega el CSV para generar matrices de confusión, curvas ROC/PR y el árbol de decisión.")
    else:
        try:
            evaluacion = entrenar_para_dashboard(df)
            nombres = list(evaluacion["modelos"])
            modelo_nombre = st.selectbox("Modelo a inspeccionar", nombres, index=nombres.index(res["ganador"]))
            modelo = evaluacion["modelos"][modelo_nombre]
            # Para el modelo ganador se usa el .joblib del notebook, así la matriz coincide
            # exactamente con las métricas exportadas (XGBoost puede variar al reentrenarse).
            if bundle and bundle.get("nombre_modelo") == modelo_nombre:
                modelo = bundle["modelo"]
            x_test, y_test = evaluacion["x_test"], evaluacion["y_test"]
            if bundle and modelo is bundle.get("modelo"):
                x_test = x_test[list(bundle["columnas"])]
            pred = modelo.predict(x_test)
            prob = modelo.predict_proba(x_test)[:, 1]
            t1, t2 = st.tabs(["Matriz de confusión", "Curvas ROC y PR"])
            with t1:
                cm = confusion_matrix(y_test, pred, labels=[0, 1])
                etiquetas_cm = ["VN · Permanece detectado", "FP · Falsa alerta", "FN · Abandono no detectado", "VP · Abandono detectado"]
                fig = go.Figure(go.Bar(x=cm.ravel(), y=etiquetas_cm, orientation="h",
                                       marker_color=[AZUL, ROJO, ROJO, AZUL], text=cm.ravel(), textposition="outside"))
                fig.update_layout(title=f"Matriz de confusión (conteos) · {modelo_nombre}", xaxis_title="Clientes", height=400)
                ui.grafica('¿Cuántos falsos negativos y falsos positivos produce el modelo seleccionado?', fig, use_container_width=True)
                tn, fp, fn, tp = cm.ravel()
                st.caption(f"Aciertos: {tn + tp:,} de {cm.sum():,}. Falsos negativos (cancela y no se detectó): "
                           f"{fn}. Falsos positivos (se marcó como riesgo y permaneció): {fp}.")
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
                    ui.grafica('¿Qué equilibrio entre detección y falsas alarmas ofrece el modelo al variar el umbral?', roc, use_container_width=True)
                with b:
                    pr = go.Figure()
                    pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=modelo_nombre,
                                            line=dict(color=ROJO, width=3)))
                    pr.update_layout(title="Precision–Recall", xaxis_title="Recall",
                                     yaxis_title="Precision", height=345)
                    ui.grafica('¿Qué equilibrio entre precisión y recall ofrece el modelo al variar el umbral?', pr, use_container_width=True)
            # El árbol no se conserva: no evalúa necesariamente el clasificador seleccionado.
            st.caption("Visualizaciones recalculadas con el CSV activo: split 80/20, semilla 42. Sus filas pueden diferir de la evaluación histórica.")
        except Exception as exc:
            st.error(f"No se pudo preparar la evaluación: {exc}")
            


# ─── 4. Campañas de retención ────────────────────────────────────────────────
elif pagina == "📣 Campañas de retención":
    ui.encabezado_pagina(
        "Campañas de retención",
        "Convierte las señales de abandono en audiencias, acciones de CRM y pruebas controladas.",
        etiqueta="Retention Intelligence",
    )
    st.caption("Señal de uso → modelo campeón → nivel de riesgo → campaña → medición")
    st.info("Modo académico: el CSV contiene resultados históricos, no una lista verificada de "
            "suscripciones activas. Las audiencias y campañas son simulaciones; no se envían "
            "mensajes ni se conceden descuentos. Antes de operar se necesita una fotografía "
            "de clientes activos anterior al abandono y un horizonte de predicción definido.")
    if df is None:
        st.warning("Carga netflix_customer_churn.csv para construir las audiencias.")
        st.stop()

    # Mismo modelo que el simulador; no entrenar con churned como variable de entrada.
    try:
        if bundle and "modelo" in bundle and "columnas" in bundle:
            modelo_camp = bundle["modelo"]
            columnas_camp = list(bundle["columnas"])
            nombre_camp = bundle.get("nombre_modelo", "Modelo exportado")
            origen_camp = "Modelo exportado por el notebook"
        elif res is not None:
            nombre_camp = res["ganador"]
            modelo_camp = entrenar_para_dashboard(df)["modelos"][nombre_camp]
            columnas_camp = COLUMNAS
            origen_camp = "Modelo reconstruido con el CSV"
        else:
            st.warning("Se requiere el modelo exportado o los resultados del notebook.")
            st.stop()
        x_camp, _ = preparar_x(df)
        # El dataset conserva el promedio observado, no el promedio reconstruido del simulador.
        num_camp = df[NUMERICAS].apply(pd.to_numeric, errors="coerce")
        validos = (np.isfinite(num_camp).all(axis=1)
                   & num_camp["watch_hours"].ge(0)
                   & num_camp["last_login_days"].ge(0)
                   & num_camp["avg_watch_time_per_day"].between(0, 24)
                   & num_camp["number_of_profiles"].between(1, 5)
                   & df["subscription_type"].isin(PLANES))
        if not validos.any():
            st.warning("No hay registros que pasen los controles básicos de calidad.")
            st.stop()
        clases_camp = list(modelo_camp.classes_)
        scores_camp = modelo_camp.predict_proba(x_camp.loc[validos, columnas_camp])[:, clases_camp.index(1)]
        if not np.isfinite(scores_camp).all() or not ((scores_camp >= 0) & (scores_camp <= 1)).all():
            raise ValueError("El modelo devolvió puntuaciones inválidas.")
    except Exception as exc:
        st.error(f"No se pudo preparar el motor de retención: {exc}")
        st.stop()

    camp = df_segmentado.loc[validos].copy()
    camp["ID cliente"] = (camp["customer_id"].astype(str) if "customer_id" in camp.columns
                          else camp.index.astype(str))
    camp["Score"] = scores_camp
    camp["Zona crítica"] = (camp["watch_hours"] < 5) & (camp["last_login_days"] > 30)
    camp["Riesgo del modelo"] = np.select([camp["Score"] >= .60, camp["Score"] >= .30],
                                         ["Alto", "Medio"], default="Bajo")
    # La zona crítica se atiende aun si el score no alcanza 0.60 (regla de negocio de la presentación).
    camp["Prioridad"] = np.where(camp["Zona crítica"], "Alto", camp["Riesgo del modelo"])
    camp["Señal preventiva"] = camp["last_login_days"].between(15, 30)
    camp["Pago por revisar"] = False
    if "payment_method" in camp.columns:
        pago = camp["payment_method"].fillna("").astype(str).str.lower()
        camp["Pago por revisar"] = pago.str.contains(r"gift|crypto|regalo|cripto", regex=True)
    camp["Campaña"] = np.select(
        [camp["Prioridad"].eq("Alto"), camp["Prioridad"].eq("Medio") | camp["Señal preventiva"]],
        ["Contacto prioritario", "Reactivación preventiva"], default="Seguimiento habitual")
    camp["Acción propuesta"] = np.select(
        [camp["Campaña"].eq("Contacto prioritario"), camp["Campaña"].eq("Reactivación preventiva")],
        ["Contacto personalizado; evaluar pausa, cambio de plan o incentivo",
         "Recomendar contenido y enviar recordatorio de reactivación"],
        default="Comunicación habitual; monitorear actividad")
    camp.loc[camp["Pago por revisar"], "Acción propuesta"] += "; facilitar renovación"
    if "favorite_genre" in camp.columns:
        genero = camp["favorite_genre"].fillna("").astype(str).str.strip()
        personalizar = camp["Campaña"].eq("Reactivación preventiva") & genero.ne("")
        camp.loc[personalizar, "Acción propuesta"] += " · Género: " + genero[personalizar]
    camp["Mensualidad"] = (pd.to_numeric(camp["monthly_fee"], errors="coerce").where(lambda s: s >= 0)
                           if "monthly_fee" in camp.columns else np.nan)
    camp["Ingreso asociado ponderado"] = camp["Mensualidad"] * camp["Score"]
    camp["Segmento"] = "Sin clúster disponible"
    if clusters_disponibles:
        mapa_camp = perfiles.set_index("cluster")["nombre"].to_dict()
        camp["Segmento"] = camp["cluster"].map(mapa_camp).fillna("Sin clúster disponible")

    st.caption(f"Motor: {nombre_camp} · {origen_camp}. Puntuaciones sin calibración adicional; "
               "no son probabilidades temporales verificadas. No se usa churned para segmentar ni priorizar.")
    st.markdown("**Reglas de activación del proyecto**")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.success("Bajo · score < 30 % · seguimiento")
    with r2:
        st.warning("Medio · 30 % ≤ score < 60 % · reactivar")
    with r3:
        st.error("Alto · score ≥ 60 % o zona crítica · intervenir")
    st.caption("Zona crítica: menos de 5 h vistas y más de 30 días sin acceso. "
               "Señal preventiva: 15–30 días sin acceso. Los límites son reglas operativas propuestas, "
               "no umbrales optimizados de clasificación. El umbral binario de 0.50 del simulador no cambia.")

    if rol == "gerencia":
        laura = st.container()
    else:
        laura, andres, sofia, medicion = st.tabs([
            "Laura · Gerencia", "Andrés · Campañas CRM", "Sofía · Calidad", "Medición · Prueba A/B"
        ])
    with laura:
        ui.seccion("Cartera de retención y prioridades")
        altos = camp[camp["Prioridad"].eq("Alto")]
        medios = camp[camp["Prioridad"].eq("Medio")]
        ingreso_alto = altos["Mensualidad"].sum(min_count=1)
        a, b, c, d = st.columns(4)
        a.metric("Clientes evaluados", f"{len(camp):,}")
        b.metric("Prioridad alta", f"{len(altos):,}", f"{len(altos)/len(camp):.1%} de la muestra")
        c.metric("Zona crítica", f"{int(camp['Zona crítica'].sum()):,}")
        d.metric("Mensualidad asociada · alta", f"${ingreso_alto:,.2f}" if pd.notna(ingreso_alto) else "N/D")
        st.caption("La mensualidad asociada no es pérdida observada ni ingreso retenido; moneda según dataset. "
                   "No hay garantía de que los clientes estén activos hoy.")
        l, r = st.columns(2)
        with l:
            dist = camp["Prioridad"].value_counts().reindex(["Bajo", "Medio", "Alto"], fill_value=0)
            fig = px.bar(x=dist.index, y=dist.values, color=dist.index,
                         color_discrete_map={"Bajo": VERDE, "Medio": AMBAR, "Alto": ROJO},
                         labels={"x": "Prioridad", "y": "Clientes"}, title="Clientes por prioridad de retención")
            fig.update_layout(showlegend=False, height=330)
            ui.grafica('¿Cuántos clientes requieren cada nivel de atención para dimensionar la campaña?', fig, use_container_width=True, key="camp_prioridad")
        with r:
            resumen = camp.groupby("subscription_type", as_index=False).agg(
                Clientes=("ID cliente", "size"),
                Alta=("Prioridad", lambda s: s.eq("Alto").sum()),
                Score_medio=("Score", "mean"),
            )
            resumen["Proporción alta"] = resumen["Alta"] / resumen["Clientes"]
            fig = px.bar(resumen, x="subscription_type", y="Alta", text_auto=True,
                         color_discrete_sequence=[ROJO], title="Prioridad alta por plan",
                         labels={"subscription_type": "Plan", "Alta": "Clientes"})
            fig.update_layout(height=330)
            ui.grafica('¿En qué plan hay más clientes de prioridad alta para asignar recursos?', fig, use_container_width=True, key="camp_plan")
            st.dataframe(resumen.style.format({"Score_medio": "{:.1%}", "Proporción alta": "{:.1%}"}),
                         hide_index=True, use_container_width=True)
        if clusters_disponibles:
            por_segmento = camp.groupby("Segmento", as_index=False).agg(
                Clientes=("ID cliente", "size"), Alta=("Prioridad", lambda s: s.eq("Alto").sum()),
                Score_medio=("Score", "mean"))
            por_segmento["Proporción alta"] = por_segmento["Alta"] / por_segmento["Clientes"]
            st.markdown("**Prioridad por clúster descriptivo**")
            st.dataframe(por_segmento.style.format({"Score_medio": "{:.1%}", "Proporción alta": "{:.1%}"}),
                         hide_index=True, use_container_width=True)
        ui.seccion("Evaluar un incentivo · escenario hipotético")
        st.write("Laura autoriza incentivos solo si el valor incremental esperado supera su costo.")
        q1, q2, q3 = st.columns(3)
        meses = q1.number_input("Meses de valor retenido", 1, 24, 3, key="camp_meses")
        eficacia = q2.slider("Abandonos evitados entre quienes cancelarían (supuesto)", 0, 100, 20,
                              key="camp_eficacia") / 100
        descuento = q3.slider("Descuento por un mes (%)", 0, 100, 50, key="camp_descuento") / 100
        costo_contacto = st.number_input("Costo por cliente contactado ($, misma moneda del dataset)",
                                         min_value=0.0, value=0.0, step=.5, key="camp_contacto")
        base_econ = altos.dropna(subset=["Mensualidad"]).copy()
        base_econ["Valor esperado"] = (base_econ["Score"] * eficacia * meses * base_econ["Mensualidad"]
                                        - descuento * base_econ["Mensualidad"] - costo_contacto)
        viables = base_econ[base_econ["Valor esperado"] > 0]
        if base_econ.empty:
            st.info("No hay clientes de prioridad alta con mensualidad válida para evaluar incentivos.")
        else:
            e1, e2 = st.columns(2)
            e1.metric("Incentivos potencialmente viables", f"{len(viables):,} de {len(base_econ):,}")
            e2.metric("Valor incremental simulado · viables", f"${viables['Valor esperado'].sum():,.2f}")
        st.caption("Fórmula ilustrativa: score × eficacia supuesta × meses × mensualidad − descuento de un mes "
                   "− costo de contacto. Supone el descuento para todo cliente tratado; no estima margen, "
                   "causalidad ni rentabilidad real. Requiere calibración y evidencia experimental.")

    if rol != "analista":
        st.caption("CRM, registros individuales y asignación A/B requieren el rol analista.")
        ui.pie(["Dataset del proyecto", "Resultados del notebook"])
        st.stop()
    with andres:
        ui.seccion("Diseñar una audiencia de campaña")
        a, b, c = st.columns(3)
        tipos_camp = a.multiselect("Tipo de campaña", ["Contacto prioritario", "Reactivación preventiva", "Seguimiento habitual"],
                                   default=["Contacto prioritario", "Reactivación preventiva"], key="camp_tipos")
        planes_camp = b.multiselect("Planes", sorted(camp["subscription_type"].unique()),
                                    default=sorted(camp["subscription_type"].unique()), key="camp_planes")
        segmentos_camp = c.multiselect("Segmentos", sorted(camp["Segmento"].unique()),
                                       default=sorted(camp["Segmento"].unique()), key="camp_segmentos")
        solo_critica = st.checkbox("Solo zona crítica", key="camp_critica")
        seleccion = camp[camp["Campaña"].isin(tipos_camp) & camp["subscription_type"].isin(planes_camp)
                         & camp["Segmento"].isin(segmentos_camp)].copy()
        if solo_critica:
            seleccion = seleccion[seleccion["Zona crítica"]].copy()
        orden = st.radio("Orden de priorización", ["Riesgo", "Ingreso ponderado por riesgo"],
                         horizontal=True, key="camp_orden")
        seleccion["_prioridad"] = seleccion["Prioridad"].map({"Alto": 2, "Medio": 1, "Bajo": 0})
        orden_col = "Score" if orden == "Riesgo" else "Ingreso asociado ponderado"
        seleccion = seleccion.sort_values(["_prioridad", orden_col, "ID cliente"],
                                          ascending=[False, False, True], na_position="last")
        capacidad = st.number_input("Capacidad máxima de la audiencia (clientes)", min_value=1,
                                     max_value=max(1, len(camp)), value=min(500, len(camp)), step=1, key="camp_capacidad")
        audiencia = seleccion.head(int(capacidad)).copy()
        st.caption(f"Campañas: {', '.join(tipos_camp) or 'Ninguna'} · Planes: {', '.join(planes_camp) or 'Ninguno'} · Segmentos: {', '.join(segmentos_camp) or 'Ninguno'} · Solo zona crítica: {'Sí' if solo_critica else 'No'}")
        if audiencia.empty:
            st.info("No hay clientes con los filtros seleccionados. Ajusta la audiencia.")
            st.caption("Audiencia seleccionada: 0 clientes")
            st.stop()
        k1, k2, k3 = st.columns(3)
        k1.metric("Clientes elegibles", f"{len(seleccion):,}")
        k2.metric("Audiencia seleccionada", f"{len(audiencia):,}")
        k3.metric("Score promedio", f"{audiencia['Score'].mean():.1%}" if len(audiencia) else "N/D")
        columnas_crm = ["ID cliente", "subscription_type", "Segmento", "last_login_days", "watch_hours",
                        "Score", "Riesgo del modelo", "Zona crítica", "Prioridad", "Campaña", "Acción propuesta",
                        "Mensualidad"]
        if "payment_method" in audiencia.columns:
            columnas_crm.append("payment_method")
        tabla_crm = audiencia[columnas_crm].rename(columns={"subscription_type": "Plan", "last_login_days": "Días sin acceso",
                                                              "watch_hours": "Horas vistas", "payment_method": "Método de pago"})
        if audiencia.empty:
            st.info("No hay clientes con los filtros seleccionados.")
        else:
            st.dataframe(tabla_crm.style.format({"Score": "{:.1%}", "Mensualidad": "{:.2f}", "Horas vistas": "{:.2f}"}, na_rep="N/D"),
                         hide_index=True, use_container_width=True)
            st.download_button("Exportar audiencia CRM (CSV)", tabla_crm.to_csv(index=False).encode("utf-8-sig"),
                               file_name="audiencia_retencion.csv", mime="text/csv", key="camp_export")
        st.caption("Los métodos de pago y el género favorito personalizan la intervención cuando existen; "
                   "no son variables de entrada del campeón Python. El CSV exportado es una propuesta, no un envío.")

    with sofia:
        ui.seccion("Calidad y condiciones para activar retención")
        a, b, c = st.columns(3)
        a.metric("Registros descartados", f"{int((~validos).sum()):,}")
        b.metric("Consumo diario > 24 h", f"{int(num_camp['avg_watch_time_per_day'].gt(24).sum()):,}")
        c.metric("Mensualidades no disponibles", f"{int(camp['Mensualidad'].isna().sum()):,}")
        st.write("Los registros con variables inválidas se excluyen de la audiencia, no del dashboard original.")
        st.dataframe(pd.DataFrame({"Control": ["Variables numéricas finitas y en rango", "Plan reconocido", "Temporalidad y estado activo", "Calibración del score", "Drift y equidad"],
                                   "Estado": ["Aplicado", "Aplicado", "Pendiente de validar con datos operativos", "Pendiente de evaluación independiente", "Requiere nuevas observaciones"]}),
                     hide_index=True, use_container_width=True)
        st.warning("No se puede comprobar con esta fotografía histórica si las variables fueron registradas "
                   "antes del abandono. No lanzar campañas reales hasta verificar fechas, clientes activos, "
                   "calidad, permisos de contacto y desempeño fuera de muestra.")
        fig = px.histogram(camp, x="Score", facet_col="Prioridad", nbins=25,
                           color_discrete_sequence=[AZUL],
                           title="Distribución de puntuaciones por prioridad (muestra histórica)")
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        ui.grafica('¿Cómo se distribuyen los scores por prioridad para revisar la cartera?', fig, use_container_width=True, key="camp_scores")

    with medicion:
        ui.seccion("Diseñar prueba controlada de retención")
        st.write("La audiencia filtrada en la pestaña de Andrés se divide aleatoriamente en tratamiento y control. "
                 "El tratamiento recibiría la campaña propuesta y el control la comunicación habitual.")
        control_pct = st.slider("Porcentaje asignado al grupo control", 10, 50, 20, step=5, key="camp_control")
        horizonte = st.number_input("Horizonte de seguimiento propuesto (días)", 7, 180, 30, key="camp_horizonte")
        semilla = st.number_input("Semilla de aleatorización", min_value=0, max_value=2147483647, value=42, key="camp_semilla")
        if len(audiencia) < 2 or not audiencia["ID cliente"].is_unique:
            st.warning("Se requieren al menos dos clientes con identificadores únicos para crear ambos grupos.")
        else:
            experimento = tabla_crm.reset_index(drop=True).copy()
            n_control = min(len(experimento) - 1, max(1, round(len(experimento) * control_pct / 100)))
            rng = np.random.default_rng(int(semilla))
            experimento["Grupo"] = "Tratamiento"
            experimento.loc[rng.choice(len(experimento), size=n_control, replace=False), "Grupo"] = "Control"
            experimento["Horizonte (días)"] = int(horizonte)
            experimento.loc[experimento["Grupo"].eq("Control"), "Acción propuesta"] = "Comunicación habitual (control)"
            a, b, c = st.columns(3)
            a.metric("Tratamiento", f"{len(experimento)-n_control:,}")
            b.metric("Control", f"{n_control:,}")
            c.metric("Seguimiento", f"{horizonte} días")
            st.dataframe(experimento.groupby("Grupo", as_index=False).agg(Clientes=("ID cliente", "size"),
                                                                                        Score_medio=("Score", "mean"))
                         .style.format({"Score_medio": "{:.1%}"}), hide_index=True, use_container_width=True)
            st.download_button("Exportar asignación A/B (CSV)", experimento.to_csv(index=False).encode("utf-8-sig"),
                               file_name="prueba_controlada_retencion.csv", mime="text/csv", key="camp_export_ab")
        st.markdown("**Indicadores que deberán medirse al finalizar el piloto**")
        st.dataframe(pd.DataFrame({"Indicador": ["Tasa de abandono por grupo", "Reducción absoluta del churn", "Ingreso neto incremental", "Costo por cliente retenido"],
                                   "Cálculo": ["Cancelaciones / clientes asignados", "Churn control − churn tratamiento", "Diferencia de ingreso por cliente × tamaño tratado − costos", "Costo incremental / abandonos evitados estimados"],
                                   "Estado": ["Pendiente de resultados reales"] * 4}), hide_index=True, use_container_width=True)
        st.caption("La asignación es una simulación reproducible, no una campaña ejecutada. Antes de iniciar "
                   "un piloto real, verificar elegibilidad, tamaño muestral, balance de grupos y fijar el horizonte. "
                   "No usar churned histórico para atribuir eficacia a una intervención que nunca ocurrió.")


# ─── 5. Predicción ───────────────────────────────────────────────────────────
elif pagina == "🎯 Predecir abandono":
    ui.encabezado_pagina("Predecir abandono",
                         "Estima la probabilidad de que un cliente cancele (churned = 1), muestra su nivel "
                         "de riesgo y explica qué variables empujaron el resultado.",
                         etiqueta="Simulador")

    # 4.1 Modelo a usar: el .joblib exportado por el notebook; si no existe, se reconstruye.
    modelo_pred, columnas_modelo, nombre_pred = None, COLUMNAS, None
    if bundle and "modelo" in bundle and "columnas" in bundle:
        modelo_pred = bundle["modelo"]
        columnas_modelo = list(bundle["columnas"])
        nombre_pred = bundle.get("nombre_modelo", "modelo exportado")
        origen = "exportado por el notebook (modelo_churn_netflix.joblib)"
    elif df is not None and res is not None:
        try:
            nombre_pred = res["ganador"]
            modelo_pred = entrenar_para_dashboard(df)["modelos"][nombre_pred]
            origen = "reconstruido con el CSV y los hiperparámetros del notebook"
        except Exception as exc:
            st.warning(f"No se pudo entrenar el modelo: {exc}")
    if modelo_pred is None:
        st.warning("Se requiere modelo_churn_netflix.joblib o el CSV original para habilitar las predicciones.")
        st.stop()
    st.caption(f"Modelo: {nombre_pred}, elegido por F1 en validación cruzada, {origen}. "
               "Estimación académica: apoya la priorización, no sustituye una decisión humana.")

    # Niveles de riesgo: mismos umbrales que las reglas de decisión del documento de SAS (sección 5).
    NIVELES = [  # (límite inferior, nombre, color, acción sugerida)
        (.60, "Alto", ROJO, "Intervención prioritaria: oferta de retención y contacto personalizado."),
        (.30, "Medio", AMBAR, "Atención preventiva: campaña de reactivación sin descuento."),
        (0.0, "Bajo", VERDE, "Seguimiento normal: mantener la estrategia actual."),
    ]

    def nivel_riesgo(p: float) -> tuple:
        return next(n for n in NIVELES if p >= n[0])

    def vector(horas: float, dias: int, perfiles: int, plan: str) -> pd.DataFrame:
        """Construye la fila de entrada con el mismo orden y codificación que el entrenamiento."""
        fila = {col: 0 for col in COLUMNAS}
        fila.update({"watch_hours": horas, "last_login_days": dias,
                     # Relación exacta en el dataset: avg = watch_hours / (last_login_days + 1)
                     "avg_watch_time_per_day": round(horas / (dias + 1), 2),
                     "number_of_profiles": perfiles, f"subscription_type_{plan}": 1})
        return pd.DataFrame([fila])[columnas_modelo]

    def probabilidad(x: pd.DataFrame) -> float:
        return float(modelo_pred.predict_proba(x)[0, 1])

    # 4.2 Entrada: perfil escrito a mano o un cliente real del dataset.
    ui.seccion("1. Datos del cliente")
    modo = st.radio("Origen del perfil", ["Escribir un perfil", "Elegir un cliente del dataset"],
                    horizontal=True, key="p_modo", disabled=df is None)
    real = None
    if modo == "Elegir un cliente del dataset" and df is not None:
        ids = df["customer_id"].astype(str).tolist() if "customer_id" in df.columns else [str(i) for i in df.index]
        elegido = st.selectbox("Cliente (customer_id)", ids, key="p_cliente",
                               help="Permite comparar la predicción con lo que realmente pasó.")
        real = df.iloc[ids.index(elegido)]
        valores = dict(horas=float(real["watch_hours"]), dias=int(real["last_login_days"]),
                       perfiles=int(real["number_of_profiles"]), plan=str(real["subscription_type"]))
        a, b, c, d = st.columns(4)
        a.metric("Horas vistas", f"{valores['horas']:.2f} h")
        b.metric("Días sin acceso", f"{valores['dias']} días")
        c.metric("Perfiles", f"{valores['perfiles']}")
        d.metric("Plan", valores["plan"])
    else:
        with st.form("perfil_cliente"):
            a, b = st.columns(2)
            with a:
                horas = st.number_input("Horas totales de visualización (h)", min_value=0.0, max_value=110.4,
                                        value=8.0, step=.5, help="En el dataset va de 0.01 a 110.4 h; mediana 8 h.")
                dias = st.slider("Días desde el último inicio de sesión", 0, 60, 30,
                                 help="En el dataset va de 0 a 60 días; mediana 30.")
            with b:
                perfiles = st.slider("Número de perfiles", 1, 5, 3)
                plan = st.selectbox("Tipo de suscripción", PLANES, index=0)
            st.form_submit_button("Calcular probabilidad de abandono", type="primary", width="stretch")
        valores = dict(horas=horas, dias=dias, perfiles=perfiles, plan=plan)

    try:
        x = vector(**valores)
        prob = probabilidad(x)
    except Exception as exc:
        st.error(f"No se pudo estimar el abandono: {exc}")
        st.stop()
    promedio = x["avg_watch_time_per_day"].iloc[0]
    limite, nivel, color, accion = nivel_riesgo(prob)

    # 4.3 Resultado
    ui.seccion("2. Resultado")
    izq, der = st.columns([1, 1.25])
    with izq:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": " %", "valueformat": ".1f"},
            title={"text": "Probabilidad de abandono"},
            gauge={"axis": {"range": [0, 100], "ticksuffix": " %"}, "bar": {"color": color, "thickness": .3},
                    "bgcolor": COLORES["tarjeta"], "bordercolor": COLORES["borde"], "borderwidth": 1,
                    "threshold": {"line": {"color": COLORES["blanco"], "width": 3}, "value": 50}}))
        gauge.update_layout(height=280, margin=dict(l=40, r=40, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)")
        ui.grafica('¿Qué riesgo estima el modelo para el perfil y qué prioridad corresponde?', gauge, use_container_width=True)
        st.caption("Bajo < 30 %, medio 30–59 %, alto ≥ 60 %. Línea blanca: umbral de 50 % del modelo.")
    with der:
        r1, r2 = st.columns(2)
        r1.metric("Nivel de riesgo", nivel)
        r2.metric("Clasificación del modelo (umbral 0.50)", "Abandona" if prob >= .5 else "Permanece")
        st.markdown(f"**Acción sugerida:** {accion}")
        st.caption(f"Horas promedio por día calculadas: {promedio:.2f} h = horas vistas / (días sin acceso + 1).")
        if real is not None:
            acierto = int(prob >= .5) == int(real["churned"])
            texto_real = "canceló" if real["churned"] == 1 else "permaneció"
            (st.success if acierto else st.error)(
                f"En la realidad este cliente **{texto_real}**. El modelo "
                f"{'acertó' if acierto else 'se equivocó'} con el umbral de 0.50.")
        st.caption("Los umbrales de 30 % y 60 % son los mismos de las reglas de decisión del proyecto: "
                   "60 % está por encima del corte de máxima separación del modelo y es donde una oferta "
                   "de retención empieza a pagarse.")

    # 4.4 Explicación local con contribuciones SHAP nativas de XGBoost
    ui.seccion("3. ¿Por qué este resultado?")
    contribuciones = None
    if hasattr(modelo_pred, "get_booster"):
        import xgboost as xgb
        valores_shap = modelo_pred.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)[0]
        contrib = pd.Series(valores_shap[:-1], index=columnas_modelo)
        nombres = {"watch_hours": "Horas vistas", "last_login_days": "Días sin acceso",
                   "avg_watch_time_per_day": "Horas promedio por día", "number_of_profiles": "Número de perfiles"}
        agrupado = {}
        for col_, v in contrib.items():
            clave = "Tipo de suscripción" if col_.startswith("subscription_type_") else nombres.get(col_, col_)
            agrupado[clave] = agrupado.get(clave, 0) + v
        contribuciones = pd.Series(agrupado).sort_values(key=abs)
    if contribuciones is None:
        st.info("La explicación por variable está disponible solo para el modelo XGBoost.")
    else:
        e1, e2 = st.columns([1.3, 1])
        with e1:
            fig = go.Figure(go.Bar(
                x=contribuciones.values, y=contribuciones.index, orientation="h",
                marker_color=[ROJO if v > 0 else GRIS for v in contribuciones.values],
                text=[f"{v:+.2f}" for v in contribuciones.values], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
            rango = max(abs(contribuciones).max() * 1.35, .5)
            fig.update_layout(title=dict(text="Contribución de cada variable (SHAP)", x=0, font=dict(size=15)),
                              xaxis=dict(title="← reduce el riesgo · aumenta el riesgo →", range=[-rango, rango],
                                         zeroline=True, zerolinecolor=COLORES["borde"]),
                              height=300, margin=dict(l=0, r=20, t=45, b=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            ui.grafica('¿Qué variables aumentan o reducen la predicción de abandono?', fig, use_container_width=True)
        with e2:
            sube = contribuciones[contribuciones > 0].sort_values(ascending=False)
            baja = contribuciones[contribuciones < 0].sort_values()
            if len(sube):
                st.markdown("**Lo que más aumenta el riesgo:** " + ", ".join(sube.index[:2]) + ".")
            if len(baja):
                st.markdown("**Lo que más lo reduce:** " + ", ".join(baja.index[:2]) + ".")
            st.caption("Valores en escala log-odds: cada barra indica cuánto mueve esa variable la predicción "
                       "respecto al cliente promedio. Rojo empuja hacia cancelar; gris hacia permanecer. "
                       "Explica al modelo, no demuestra causas.")

    # 4.5 Escenarios: cómo cambia el riesgo si cambia el comportamiento
    ui.seccion("4. ¿Qué pasaría si...?")
    rango_dias = list(range(0, 61, 2))
    rango_horas = [round(h, 1) for h in np.linspace(0, 40, 41)]
    curva_dias = [probabilidad(vector(valores["horas"], d, valores["perfiles"], valores["plan"])) for d in rango_dias]
    curva_horas = [probabilidad(vector(h, valores["dias"], valores["perfiles"], valores["plan"])) for h in rango_horas]

    def curva(xs, ys, actual_x, titulo, eje_x):
        fig = go.Figure()
        for limite_riesgo in (30, 60):
            fig.add_hline(y=limite_riesgo, line_dash="dot", line_color=GRIS,
                          annotation_text=f"Umbral {limite_riesgo} %")
        fig.add_trace(go.Scatter(x=xs, y=[v * 100 for v in ys], mode="lines", line=dict(color=ROJO, width=3),
                                 hovertemplate=eje_x + ": %{x}<br>Riesgo: %{y:.1f} %<extra></extra>"))
        fig.add_trace(go.Scatter(x=[actual_x], y=[prob * 100], mode="markers", marker=dict(size=12, color=COLORES["blanco"]),
                                 hovertemplate="Cliente actual: %{y:.1f} %<extra></extra>"))
        fig.update_layout(title=dict(text=titulo, x=0, font=dict(size=15)), showlegend=False, height=300,
                          xaxis_title=eje_x, yaxis=dict(title="Probabilidad de abandono (%)", range=[0, 100]),
                          margin=dict(l=0, r=10, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        return fig

    w1, w2 = st.columns(2)
    with w1:
        ui.grafica('¿Cómo cambia el riesgo estimado si varían los días sin acceso?', curva(rango_dias, curva_dias, valores["dias"], "Si cambian los días sin acceso",
                              "Días desde el último acceso"), use_container_width=True)
    with w2:
        ui.grafica('¿Cómo cambia el riesgo estimado si varían las horas vistas?', curva(rango_horas, curva_horas, min(valores["horas"], 40), "Si cambian las horas vistas",
                              "Horas vistas (h)"), use_container_width=True)
    prob_hoy = probabilidad(vector(valores["horas"], 0, valores["perfiles"], valores["plan"]))
    st.write(f"Si este cliente volviera a entrar hoy (0 días sin acceso) y conservara sus demás datos, su riesgo "
             f"estimado pasaría de **{prob:.1%}** a **{prob_hoy:.1%}**.")
    st.caption("Cada curva cambia una sola variable y deja fijas las demás; las horas promedio por día se "
               "recalculan con la misma fórmula. El punto blanco es el cliente actual (en horas se muestra hasta 40 h). "
               "Son escenarios del modelo, no efectos garantizados de una acción.")

ui.pie(["notebook netflix_churned.ipynb", "datos/netflix_validado.csv (derivado de la fuente académica)",
        f"datos al {fecha_datos:%d/%m/%Y}" if fecha_datos else "fecha no disponible"])
