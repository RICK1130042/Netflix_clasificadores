import streamlit as st
import pandas as pd

# 1. Configuración inicial de la página
st.set_page_config(page_title="Netflix Dashboards", layout="wide", initial_sidebar_state="expanded")

# 2. Navegación lateral (Simulador de Páginas)
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
st.sidebar.title("Navegación")
st.sidebar.markdown("Elige qué dashboard quieres ver:")
pagina = st.sidebar.radio("", ["🍿 Mi Resumen: Juan", "🍷 Mi Resumen: Mariana", "📊 Retención (Negocio)"])

st.sidebar.markdown("---")

# ==========================================
# PÁGINA 1: JUAN
# ==========================================
if pagina == "🍿 Mi Resumen: Juan":
    st.title("Tu Año en Netflix 🍿")
    st.markdown("### Juan, aquí están tus hábitos de visualización.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Horas", "1,240 h", "Nivel Maratón")
    col2.metric("Películas Vistas", "185")
    col3.metric("Series Iniciadas", "32")
    
    st.write("---")
    st.subheader("Tus Favoritos del Año")
    col4, col5 = st.columns(2)
    with col4:
        st.info("🎬 **Película más vista:** El Irlandés (La repetiste 3 veces)")
        st.success("✅ **Completitud de series:** 85% Terminadas")
        st.write("🧛‍♂️ **Género favorito:** Terror / Sci-Fi")
    with col5:
        st.info("📺 **Serie más vista:** Stranger Things (Temp 4 en un finde)")
        st.error("💔 **Serie más abandonada:** The Witcher (Temp 3)")
        st.write("⭐ **Actor más visto:** Pedro Pascal")
        
    st.write("---")
    st.subheader("Tu Personalidad de Espectador")
    st.markdown("> **🛋️ El Maratónico Imparable:** Si empiezas una serie, el mundo deja de existir hasta que ves los créditos finales. Duermes poco, pero sabes mucho de cultura pop. Tu día favorito de maratón son los **Domingos**.")

# ==========================================
# PÁGINA 2: MARIANA
# ==========================================
elif pagina == "🍷 Mi Resumen: Mariana":
    st.title("Tu Año en Netflix 🍷")
    st.markdown("### Mariana, tu estilo es único. Mira tus datos.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Horas", "315 h", "Modo Relax")
    col2.metric("Películas Vistas", "45")
    col3.metric("Series Iniciadas", "12")
    
    st.write("---")
    st.subheader("Tus Favoritos del Año")
    col4, col5 = st.columns(2)
    with col4:
        st.info("🎬 **Película más vista:** Orgullo y Prejuicio (Tu lugar seguro)")
        st.warning("⚠️ **Completitud de series:** 40% Terminadas")
        st.write("💕 **Género favorito:** Romance / Comedia")
    with col5:
        st.info("📺 **Serie más vista:** Bridgerton (Reina del chisme)")
        st.error("💔 **Serie más abandonada:** The Crown (Temp 5 - demasiada política)")
        st.write("⭐ **Actriz más vista:** Nicola Coughlan")
        
    st.write("---")
    st.subheader("Tu Personalidad de Espectador")
    st.markdown("> **🥂 La Romántica Casual:** Ves Netflix para desconectar. Te gusta el drama romántico, una copita y relajarte. Si te aburre, cambias de serie sin piedad. Tu día favorito de maratón son los **Viernes**.")

# ==========================================
# PÁGINA 3: NEGOCIO (CHURN)
# ==========================================
elif pagina == "📊 Retención (Negocio)":
    st.title("Análisis de Abandono (Churn) 📊")
    st.markdown("Dashboard para Toma de Decisiones - Dataset Kaggle")

    try:
        # Leemos el archivo CSV
        df = pd.read_csv("netflix_customer_churn.csv")
        
        # Variables calculadas
        df['annual_fee'] = df['monthly_fee'] * 12
        
        def categorizar_edad(edad):
            if edad < 25: return '1. Menores de 25'
            elif edad <= 40: return '2. 25 a 40 años'
            elif edad <= 55: return '3. 41 a 55 años'
            else: return '4. Mayores de 55'
            
        df['age_group'] = df['age'].apply(categorizar_edad)

        # Filtro exclusivo de esta página
        st.sidebar.subheader("Filtros de Retención")
        generos = ["Todos"] + list(df['gender'].unique())
        genero_seleccionado = st.sidebar.selectbox("Filtro por Género (Solo Negocio)", generos)
        
        if genero_seleccionado != "Todos":
            df = df[df['gender'] == genero_seleccionado]

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Total de Usuarios", value=f"{len(df):,}")
        with col2:
            churn_total = len(df[df['churned'] == 1]) 
            tasa_churn = (churn_total / len(df)) * 100 if len(df) > 0 else 0
            st.metric(label="Tasa de Abandono", value=f"{tasa_churn:.1f}%")
        with col3:
            promedio_mensual = df['monthly_fee'].mean()
            st.metric(label="Ingreso Mens. Prom.", value=f"${promedio_mensual:.2f}")
        with col4:
            promedio_anual = df['annual_fee'].mean()
            st.metric(label="Ingreso Anual Prom.", value=f"${promedio_anual:.2f}")

        # Gráficas
        st.write("---")
        col_graf1, col_graf2 = st.columns(2)
        
        with col_graf1:
            st.subheader("Abandono por Tipo de Suscripción")
            # Contar churn por plan
            churn_df = df[df['churned'] == 1]
            plan_counts = churn_df['subscription_type'].value_counts()
            st.bar_chart(plan_counts, color="#E50914") 

        with col_graf2:
            st.subheader("Abandono por Grupo de Edad")
            edad_churn = churn_df['age_group'].value_counts().sort_index()
            st.bar_chart(edad_churn, color="#555555")

    except FileNotFoundError:
        st.error("⚠️ No encuentro el archivo 'netflix_customer_churn.csv'. Asegúrate de que esté en la misma carpeta.")
    except Exception as e:
        st.error(f"Ocurrió un error al leer los datos: {e}")