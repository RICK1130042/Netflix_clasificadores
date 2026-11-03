
"""Dashboard interactivo de clasificación de abandono en Netflix."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# =========================================================
# 1. CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Netflix | Churn Analytics",
    page_icon="🎬",
    layout="wide"
)

# Ubicación de los archivos del proyecto
CARPETA_APP = Path(__file__).resolve().parent

BASE = (
    CARPETA_APP.parent
    if CARPETA_APP.name == "dashboard"
    else CARPETA_APP
)

DATASET = BASE / "netflix_customer_churn.csv"
METRICAS = BASE / "resultados_modelos.csv"
EVALUACION = BASE / "evaluacion_modelos.csv"
IMPORTANCIAS = BASE / "importancia_variables.csv"

ARBOL = BASE / "visualizaciones" / "arbol_decision.png"

# Paleta de colores inspirada en Netflix
ROJO = "#E50914"
FONDO = "#0E1117"
PANEL = "#171C26"
GRIS = "#AEB7C5"

COLORES = {
    "Permanece": "#8093AA",
    "Canceló": ROJO
}

PALETA = [
    "#F43F5E",
    "#38BDF8",
    "#FBBF24",
    "#34D399",
    "#A78BFA",
    "#FB923C"
]

METRICAS_MODELOS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC",
    "Average Precision"
]

ORDEN = [
    "Naive Bayes",
    "KNN",
    "Decision Tree",
    "Random Forest",
    "MLP",
    "SVM"
]

# =========================================================
# 2. ESTILO VISUAL
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #0E1117;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    letter-spacing: -0.03em;
}

[data-testid="stMetric"] {
    background: #171C26;
    padding: 17px;
    border-radius: 14px;
    border: 1px solid #293140;
}

[data-testid="stMetricLabel"] {
    color: #B8C1CF;
}

.hero {
    background: linear-gradient(
        110deg,
        #2C111A,
        #171C26 72%
    );
    border: 1px solid #54313A;
    padding: 26px 30px;
    border-radius: 18px;
    margin-bottom: 18px;
}

.hero h1 {
    margin: 3px 0 8px;
    color: #FFFFFF;
    font-size: clamp(25px, 3vw, 39px);
}

.hero p {
    color: #C7D0DD;
    margin: 0;
}

.eyebrow {
    color: #FF6B75;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.17em;
}

[data-testid="stTabs"] button[role="tab"] {
    font-weight: 650;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. FUNCIONES AUXILIARES
# =========================================================

@st.cache_data(show_spinner=False)
def cargar_csv(ruta):
    return pd.read_csv(ruta)


def estilo(fig, altura=410):
    """Aplica un estilo oscuro y uniforme a Plotly."""

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=FONDO,
        plot_bgcolor=PANEL,
        font=dict(color="#E8EDF5", size=12),
        height=altura,
        margin=dict(l=30, r=20, t=55, b=50),
        legend=dict(
            orientation="h",
            y=-0.24,
            x=0
        )
    )

    return fig


def clave_modelo(nombre):
    """Organiza los clasificadores en un orden fijo."""

    for i, base in enumerate(ORDEN):
        if nombre == base or nombre.startswith(base + " ("):
            return i

    return len(ORDEN)


# =========================================================
# 4. CARGAR Y VALIDAR ARCHIVOS
# =========================================================

if not DATASET.exists() or not METRICAS.exists():

    st.error(
        "Faltan los CSV principales en la carpeta "
        "Netflix_clasificadores."
    )

    st.code(f"{DATASET}\n{METRICAS}")

    st.stop()

try:

    df = cargar_csv(DATASET).copy()
    resultados = cargar_csv(METRICAS).copy()

except Exception as exc:

    st.error(f"No fue posible leer los archivos CSV: {exc}")
    st.stop()


if (
    "churned" not in df
    or "Modelo" not in resultados
    or "F1-Score" not in resultados
):

    st.error(
        "Falta churned en el dataset o "
        "Modelo/F1-Score en las métricas."
    )

    st.stop()


# Convertimos las métricas a valores numéricos
for columna in METRICAS_MODELOS:

    if columna in resultados:

        resultados[columna] = pd.to_numeric(
            resultados[columna],
            errors="coerce"
        )


df["churned"] = pd.to_numeric(
    df["churned"],
    errors="coerce"
)

df["Estado"] = df["churned"].map({
    0: "Permanece",
    1: "Canceló"
})


# Cargamos las predicciones exportadas desde el notebook
evaluacion = None

if EVALUACION.exists():

    tmp = cargar_csv(EVALUACION).copy()

    columnas = {
        "Modelo",
        "y_real",
        "y_pred",
        "score"
    }

    if columnas.issubset(tmp.columns):

        for col in ["y_real", "y_pred", "score"]:

            tmp[col] = pd.to_numeric(
                tmp[col],
                errors="coerce"
            )

        evaluacion = tmp.dropna(
            subset=["Modelo", "y_real", "y_pred", "score"]
        )


# =========================================================
# 5. ENCABEZADO
# =========================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">PROYECTO DE MACHINE LEARNING</div>
    <h1>🎬 Netflix · Customer Churn Analytics</h1>
    <p>
        ¿Qué patrones se relacionan con el abandono
        y qué clasificador logra identificarlo mejor?
    </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Dataset de Kaggle · Clasificación supervisada · "
    "Resultados obtenidos previamente en el notebook"
)


# =========================================================
# 6. NAVEGACIÓN
# =========================================================

tab_resumen, tab_eda, tab_modelos, tab_interpretacion = st.tabs([
    "📊 Resumen",
    "🔎 Exploración y correlación",
    "🤖 Clasificadores",
    "🌳 Interpretación"
])


# =========================================================
# PESTAÑA 1: RESUMEN
# =========================================================

with tab_resumen:

    total = len(df)
    cancelaron = int(df["churned"].eq(1).sum())

    mejor = resultados.loc[
        resultados["F1-Score"].idxmax()
    ]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Usuarios analizados",
        f"{total:,}"
    )

    c2.metric(
        "Usuarios que cancelaron",
        f"{cancelaron:,}"
    )

    c3.metric(
        "Tasa de abandono",
        f"{cancelaron / total:.1%}"
    )

    c4.metric(
        "Mejor F1-Score",
        f"{mejor['F1-Score']:.2%}",
        help=str(mejor["Modelo"])
    )

    st.markdown("### ¿Qué significa Churn?")

    st.write(
        "**Churned = 0:** el usuario permanece en Netflix. "
        "**Churned = 1:** el usuario abandonó o canceló "
        "su suscripción."
    )

    # Distribución de las clases
    conteos = (
        df["Estado"]
        .value_counts()
        .reindex(["Permanece", "Canceló"], fill_value=0)
        .reset_index()
    )

    conteos.columns = ["Estado", "Usuarios"]

    fig = px.bar(
        conteos,
        x="Estado",
        y="Usuarios",
        color="Estado",
        text="Usuarios",
        color_discrete_map=COLORES,
        title="Distribución de clientes"
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        showlegend=False,
        yaxis_range=[
            0,
            conteos["Usuarios"].max() * 1.16
        ]
    )

    st.plotly_chart(
        estilo(fig),
        use_container_width=True
    )

    st.info(
        "**Interpretación:** las dos clases están "
        "prácticamente equilibradas. El notebook utilizó "
        "4,000 usuarios para entrenamiento y 1,000 "
        "para prueba."
    )

    st.caption(
        "Los resultados de este dataset no garantizan "
        "la misma precisión para cancelaciones futuras reales."
    )


# =========================================================
# PESTAÑA 2: EXPLORACIÓN Y CORRELACIÓN
# =========================================================

with tab_eda:

    st.subheader("Exploración del comportamiento de los clientes")

    st.write(
        "Comparamos las características de los usuarios "
        "que permanecieron y de aquellos que cancelaron."
    )

    # -----------------------------------------------------
    # Distribución según categorías
    # -----------------------------------------------------

    categorias = {
        "Tipo de suscripción": "subscription_type",
        "Dispositivo": "device",
        "Método de pago": "payment_method",
        "Región": "region"
    }

    categorias = {
        k: v
        for k, v in categorias.items()
        if v in df.columns
    }

    if categorias:

        etiqueta = st.selectbox(
            "Comparar tasa de abandono por",
            list(categorias)
        )

        col = categorias[etiqueta]

        grupo = (
            df.groupby(col)
            .agg(
                Usuarios=("churned", "size"),
                Churn=("churned", "mean")
            )
            .reset_index()
        )

        grupo["Churn"] *= 100

        fig = px.bar(
            grupo,
            x=col,
            y="Churn",
            hover_data=["Usuarios"],
            text="Churn",
            color_discrete_sequence=[ROJO],
            title=f"Tasa de abandono según {etiqueta.lower()}"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title=etiqueta,
            yaxis_title="Clientes que cancelaron (%)",
            yaxis_range=[0, 105]
        )

        st.plotly_chart(
            estilo(fig),
            use_container_width=True
        )

        st.caption(
            "La gráfica muestra el porcentaje de abandono "
            "dentro de cada categoría, no la cantidad total "
            "de clientes."
        )

    # -----------------------------------------------------
    # Variables numéricas
    # -----------------------------------------------------

    num = [
        c for c in [
            "watch_hours",
            "last_login_days",
            "avg_watch_time_per_day",
            "age",
            "number_of_profiles"
        ]
        if c in df.columns
    ]

    if num:

        a, b = st.columns([2, 1])

        with a:
            variable = st.selectbox(
                "Característica numérica",
                num
            )

        with b:
            tipo = st.radio(
                "Gráfico",
                ["Boxplot", "Histograma"],
                horizontal=True
            )

        if tipo == "Boxplot":

            fig = px.box(
                df,
                x="Estado",
                y=variable,
                color="Estado",
                color_discrete_map=COLORES,
                title=f"{variable} según estado del cliente"
            )

        else:

            fig = px.histogram(
                df,
                x=variable,
                color="Estado",
                barmode="overlay",
                color_discrete_map=COLORES,
                opacity=0.67,
                nbins=30,
                title=f"Distribución de {variable}"
            )

        fig.update_layout(
            showlegend=(tipo == "Histograma")
        )

        st.plotly_chart(
            estilo(fig),
            use_container_width=True
        )

        medianas = (
            df.groupby("Estado")[variable]
            .median()
        )

        st.caption(
            f"Mediana — Permanece: "
            f"{medianas.get('Permanece', 0):.2f} | "
            f"Canceló: {medianas.get('Canceló', 0):.2f}"
        )

    # -----------------------------------------------------
    # MAPA DE CALOR DE CORRELACIONES
    # -----------------------------------------------------

    st.divider()

    st.subheader("Mapa de calor de correlaciones")

    st.write(
        "Este gráfico muestra qué tan relacionadas están "
        "las variables mediante el coeficiente de Pearson."
    )

    st.markdown(
        """
        - **Correlación positiva (+):** las variables tienden
          a aumentar juntas.
        - **Correlación negativa (−):** cuando una aumenta,
          la otra tiende a disminuir.
        - **Correlación cercana a 0:** existe poca relación
          lineal entre ambas.
        """
    )

    vista = st.radio(
        "Vista del mapa de calor",
        [
            "Variables numéricas originales",
            "Después de One-Hot Encoding"
        ],
        horizontal=True
    )

    if vista == "Variables numéricas originales":

        campos = [
            "age",
            "watch_hours",
            "last_login_days",
            "monthly_fee",
            "number_of_profiles",
            "avg_watch_time_per_day",
            "churned"
        ]

        campos = [
            c for c in campos if c in df.columns
        ]

        correlacion = df[campos].corr(
            numeric_only=True
        )

        alto = 540

        st.caption(
            "Este mapa incluye monthly_fee para la "
            "exploración, aunque no se utilizó "
            "para entrenar los modelos."
        )

    else:

        features = [
            "age",
            "subscription_type",
            "watch_hours",
            "last_login_days",
            "device",
            "payment_method",
            "number_of_profiles",
            "avg_watch_time_per_day",
            "churned"
        ]

        base_corr = df[
            [c for c in features if c in df]
        ].copy()

        categoricas = [
            c for c in [
                "subscription_type",
                "device",
                "payment_method"
            ]
            if c in base_corr
        ]

        codificado = pd.get_dummies(
            base_corr,
            columns=categoricas,
            dtype=int
        )

        correlacion = codificado.corr(
            numeric_only=True
        )

        alto = 780

        st.caption(
            "Las variables categóricas se transformaron "
            "en columnas binarias mediante One-Hot Encoding."
        )

    # Heatmap interactivo con Plotly
    fig = go.Figure(
        go.Heatmap(
            z=correlacion.values,
            x=correlacion.columns.tolist(),
            y=correlacion.index.tolist(),
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdBu_r",
            colorbar=dict(title="Pearson"),
            hovertemplate=(
                "%{y} vs %{x}"
                "<br>Correlación = %{z:.3f}"
                "<extra></extra>"
            )
        )
    )

    # Anotaciones numéricas para el mapa pequeño
    if len(correlacion) <= 8:

        for i, fila in enumerate(correlacion.index):

            for j, columna in enumerate(correlacion.columns):

                valor = correlacion.iloc[i, j]

                fig.add_annotation(
                    x=columna,
                    y=fila,
                    text=f"{valor:.2f}",
                    showarrow=False,
                    font=dict(
                        size=11,
                        color=(
                            "white"
                            if abs(valor) > 0.55
                            else "#101820"
                        )
                    )
                )

    fig.update_layout(
        xaxis=dict(
            tickangle=-45,
            side="bottom"
        ),
        yaxis=dict(
            autorange="reversed"
        )
    )

    st.plotly_chart(
        estilo(fig, alto),
        use_container_width=True
    )

    st.info(
        "**Hallazgo del notebook:** "
        "`watch_hours` tiene una correlación negativa "
        "con churn (r ≈ −0.480), mientras que "
        "`last_login_days` presenta una correlación "
        "positiva (r ≈ +0.472)."
    )

    st.caption(
        "Estas correlaciones tienen magnitud moderada. "
        "No demuestran que una variable provoque la cancelación."
    )


# =========================================================
# PESTAÑA 3: CLASIFICADORES
# =========================================================

with tab_modelos:

    st.subheader("Comparación de clasificadores")

    st.write(
        "Comparamos seis algoritmos utilizando el mismo "
        "conjunto de prueba. La evaluación considera "
        "diferentes métricas."
    )

    definiciones = {

        "Accuracy":
            "Proporción de predicciones correctas "
            "entre todos los usuarios.",

        "Precision":
            "De los usuarios señalados como Churn, "
            "cuántos realmente cancelaron.",

        "Recall":
            "De todos los usuarios que cancelaron, "
            "cuántos fueron detectados.",

        "F1-Score":
            "Equilibrio entre Precision y Recall "
            "para la clase Churn.",

        "ROC-AUC":
            "Capacidad del modelo para distinguir "
            "las dos clases a distintos umbrales.",

        "Average Precision":
            "Resumen del rendimiento en la curva "
            "Precision-Recall."
    }

    disponibles = [
        m for m in METRICAS_MODELOS
        if m in resultados.columns
    ]

    metrica = st.selectbox(
        "Métrica a comparar",
        disponibles,
        index=(
            3 if "F1-Score" in disponibles else 0
        )
    )

    st.caption(definiciones[metrica])

    ordenado = resultados.sort_values(
        metrica,
        ascending=False
    ).copy()

    ordenado["Valor"] = ordenado[metrica] * 100

    fig = px.bar(
        ordenado,
        x="Modelo",
        y="Valor",
        text="Valor",
        title=f"Clasificadores según {metrica}",
        color_discrete_sequence=[ROJO]
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="Resultado (%)",
        xaxis_title="",
        yaxis_range=[0, 105]
    )

    st.plotly_chart(
        estilo(fig),
        use_container_width=True
    )

    st.dataframe(
        resultados.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "**Interpretación:** Random Forest obtuvo el "
        "mejor F1-Score y ROC-AUC. Naive Bayes destacó "
        "por su Recall, aunque tuvo más falsos positivos."
    )

    # -----------------------------------------------------
    # MATRICES DE CONFUSIÓN
    # -----------------------------------------------------

    st.divider()

    st.subheader("Matrices de confusión — Todos los modelos")

    st.write(
        "**Diagonal:** predicciones correctas. "
        "**Falsos positivos:** predijo cancelación, "
        "pero el usuario permaneció. "
        "**Falsos negativos:** el usuario canceló "
        "y el modelo no lo detectó."
    )

    if evaluacion is None:

        st.warning(
            "Falta evaluacion_modelos.csv. "
            "Ejecuta la celda de exportación del notebook."
        )

    else:

        nombres = sorted(
            evaluacion["Modelo"].unique().tolist(),
            key=clave_modelo
        )

        if len(nombres) != 6:

            st.warning(
                f"Se encontraron {len(nombres)} modelos. "
                "Se esperaban seis."
            )

        # Dos columnas, tres filas para seis modelos
        for inicio in range(0, len(nombres), 2):

            columnas = st.columns(2)

            for contenedor, nombre in zip(
                columnas,
                nombres[inicio:inicio + 2]
            ):

                with contenedor:

                    sub = evaluacion[
                        evaluacion["Modelo"] == nombre
                    ]

                    matriz = confusion_matrix(
                        sub["y_real"].astype(int),
                        sub["y_pred"].astype(int),
                        labels=[0, 1]
                    )

                    etiquetas = [
                        "0 · Permanece",
                        "1 · Canceló"
                    ]

                    fig = go.Figure(
                        go.Heatmap(
                            z=matriz,
                            x=etiquetas,
                            y=etiquetas,
                            colorscale=[
                                [0, "#FBE8E5"],
                                [1, "#8F0A17"]
                            ],
                            showscale=False,
                            zmin=0,
                            zmax=500,
                            hovertemplate=(
                                "Real %{y}"
                                "<br>Predicción %{x}"
                                "<br>Usuarios: %{z}"
                                "<extra></extra>"
                            )
                        )
                    )

                    # Mostramos números dentro de la matriz
                    for i in range(2):

                        for j in range(2):

                            fig.add_annotation(
                                x=etiquetas[j],
                                y=etiquetas[i],
                                text=str(matriz[i, j]),
                                showarrow=False,
                                font=dict(
                                    size=19,
                                    color=(
                                        "#FFFFFF"
                                        if matriz[i, j] >= 250
                                        else "#312328"
                                    )
                                )
                            )

                    fig.update_layout(
                        title=f"Matriz — {nombre}",
                        xaxis_title="Predicción",
                        yaxis_title="Valor real",
                        yaxis_autorange="reversed"
                    )

                    st.plotly_chart(
                        estilo(fig, 325),
                        use_container_width=True,
                        key=f"cm_{nombre}"
                    )

                    tn, fp, fn, tp = matriz.ravel()

                    st.caption(
                        f"VN: {tn} · FP: {fp} · "
                        f"FN: {fn} · VP: {tp} "
                        f"| Aciertos: {tn + tp:,}"
                    )

        # -------------------------------------------------
        # CURVAS ROC Y PRECISION-RECALL
        # -------------------------------------------------

        st.divider()

        st.subheader("Curvas ROC y Precision-Recall")

        st.write(
            "**ROC:** cuanto más cerca esté la curva "
            "de la esquina superior izquierda, mejor "
            "es la discriminación del modelo."
        )

        st.write(
            "**Precision-Recall:** permite observar "
            "el equilibrio entre detectar cancelaciones "
            "y evitar falsas alarmas."
        )

        # Este filtro solamente modifica las curvas
        seleccion = st.multiselect(
            "Modelos en las curvas",
            nombres,
            default=nombres
        )

        if seleccion:

            roc_fig = go.Figure()
            pr_fig = go.Figure()

            for nombre in seleccion:

                sub = evaluacion[
                    evaluacion["Modelo"] == nombre
                ]

                y_real = (
                    sub["y_real"]
                    .astype(int)
                    .to_numpy()
                )

                scores = sub["score"].to_numpy()

                if len(set(y_real)) < 2:
                    continue

                fpr, tpr, _ = roc_curve(
                    y_real,
                    scores
                )

                precision, recall, _ = (
                    precision_recall_curve(
                        y_real,
                        scores
                    )
                )

                auc_val = roc_auc_score(
                    y_real,
                    scores
                )

                ap_val = average_precision_score(
                    y_real,
                    scores
                )

                color = PALETA[
                    clave_modelo(nombre) % len(PALETA)
                ]

                roc_fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        name=f"{nombre} ({auc_val:.3f})",
                        line=dict(
                            color=color,
                            width=2.5
                        ),
                        mode="lines"
                    )
                )

                pr_fig.add_trace(
                    go.Scatter(
                        x=recall,
                        y=precision,
                        name=f"{nombre} ({ap_val:.3f})",
                        line=dict(
                            color=color,
                            width=2.5
                        ),
                        mode="lines"
                    )
                )

            # Referencia de clasificador aleatorio
            roc_fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    showlegend=False,
                    line=dict(
                        color=GRIS,
                        dash="dash"
                    ),
                    mode="lines"
                )
            )

            roc_fig.update_layout(
                title="Curva ROC · AUC",
                xaxis_title="Tasa de falsos positivos",
                yaxis_title="Tasa de verdaderos positivos"
            )

            pr_fig.update_layout(
                title="Precision-Recall · AP",
                xaxis_title="Recall",
                yaxis_title="Precision"
            )

            izquierda, derecha = st.columns(2)

            with izquierda:

                st.plotly_chart(
                    estilo(roc_fig, 480),
                    use_container_width=True
                )

            with derecha:

                st.plotly_chart(
                    estilo(pr_fig, 480),
                    use_container_width=True
                )

            st.caption(
                "ROC-AUC y Average Precision no son "
                "porcentajes de predicciones correctas. "
                "SVM puede utilizar puntuaciones de decisión "
                "en lugar de probabilidades."
            )

        else:

            st.info(
                "Selecciona uno o más clasificadores "
                "para visualizar las curvas."
            )


# =========================================================
# PESTAÑA 4: INTERPRETACIÓN Y ÁRBOL
# =========================================================

with tab_interpretacion:

    st.subheader(
        "¿Qué aprendieron los modelos basados en árboles?"
    )

    st.write(
        "La importancia de variables permite entender "
        "qué características utilizaron los modelos "
        "para distinguir clientes que permanecen "
        "y clientes que cancelaron."
    )

    # -----------------------------------------------------
    # IMPORTANCIA DE VARIABLES
    # -----------------------------------------------------

    if IMPORTANCIAS.exists():

        imp = cargar_csv(IMPORTANCIAS).copy()

        if {
            "Modelo",
            "Variable",
            "Importancia"
        }.issubset(imp.columns):

            opciones = (
                imp["Modelo"]
                .dropna()
                .unique()
                .tolist()
            )

            elegido = st.radio(
                "Importancia de variables del modelo",
                opciones,
                horizontal=True
            )

            sub = imp[
                imp["Modelo"] == elegido
            ].nlargest(
                12,
                "Importancia"
            )

            sub = sub.sort_values("Importancia")

            fig = px.bar(
                sub,
                x="Importancia",
                y="Variable",
                orientation="h",
                color_discrete_sequence=[ROJO],
                title=f"Importancias — {elegido}"
            )

            fig.update_layout(
                xaxis_title="Importancia relativa",
                yaxis_title=""
            )

            st.plotly_chart(
                estilo(fig, 480),
                use_container_width=True
            )

            principal = (
                imp[imp["Modelo"] == elegido]
                .sort_values(
                    "Importancia",
                    ascending=False
                )
                .iloc[0]
            )

            st.info(
                f"**Variable de mayor importancia "
                f"en {elegido}:** "
                f"`{principal['Variable']}` "
                f"({principal['Importancia']:.1%})."
            )

            st.caption(
                "La importancia mide la contribución de "
                "una característica al modelo. "
                "No demuestra causalidad."
            )

        else:

            st.warning(
                "El archivo importancia_variables.csv "
                "no contiene las columnas esperadas."
            )

    else:

        st.info(
            "Falta importancia_variables.csv. "
            "Se genera desde el notebook."
        )

    # -----------------------------------------------------
    # ÁRBOL DE DECISIÓN
    # -----------------------------------------------------

    st.divider()

    st.subheader("Árbol de decisión — Primeros niveles")

    st.write(
        "Un árbol de decisión clasifica mediante preguntas. "
        "Cada nodo divide a los usuarios utilizando una "
        "característica y un valor de referencia."
    )

    st.write(
        "En nuestro notebook, la primera división "
        "utiliza `avg_watch_time_per_day`, es decir, "
        "el tiempo promedio de visualización diaria."
    )

    if ARBOL.exists():

        st.image(
            str(ARBOL),
            use_container_width=True
        )

        st.markdown(
            """
            **Cómo interpretar el árbol:**

            - **Gini:** indica qué tan mezcladas están
              las clases dentro del nodo.
            - **Samples:** cantidad de usuarios que
              llegan a ese nodo.
            - **Value:** distribución de las clases.
            - **Class:** clase predominante del nodo.
            """
        )

        st.caption(
            "El gráfico muestra los primeros niveles "
            "del árbol entrenado en el notebook."
        )

    else:

        st.warning(
            "Falta visualizaciones/arbol_decision.png. "
            "Ejecuta el exportador del notebook."
        )

    # -----------------------------------------------------
    # CONCLUSIÓN
    # -----------------------------------------------------

    st.divider()

    st.subheader("Conclusión del proyecto")

    st.success(
        "**Random Forest obtuvo el mejor desempeño "
        "general** en nuestro experimento, considerando "
        "Accuracy, F1-Score y ROC-AUC."
    )

    st.write(
        "Encontramos que el consumo de contenido y "
        "los días de inactividad son características "
        "relacionadas con el abandono de los usuarios."
    )

    st.write(
        "Los resultados respaldan nuestra hipótesis "
        "para el conjunto de datos analizado. "
        "Sin embargo, antes de utilizar el modelo "
        "para anticipar cancelaciones reales sería "
        "necesario validarlo con nuevos datos."
    )
