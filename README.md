# 🎬 Netflix | Customer Churn Analytics Ricardo Villarreal 

### 📊 Data Analytics & Machine Learning Project

## 📌 Descripción del proyecto

Este proyecto tiene como objetivo analizar el comportamiento de los usuarios de **Netflix**, identificar patrones relacionados con la cancelación de suscripciones (*Customer Churn*) y desarrollar modelos de Machine Learning que permitan clasificar a los clientes según su permanencia o abandono del servicio.

Se utiliza el dataset **Netflix Customer Churn and Engagement**, obtenido de Kaggle, que contiene **5,000 registros y 14 variables originales**.

## 🎯 Objetivos e hipótesis

- Realizar un análisis exploratorio de datos (EDA) para comprender el comportamiento de los suscriptores.
- Identificar y seleccionar las variables más relevantes para el análisis predictivo.
- Analizar si una menor actividad de visualización y mayores periodos de inactividad se relacionan con el abandono.
- Comparar diferentes modelos de clasificación supervisada.
- Evaluar la capacidad de **Random Forest** para identificar usuarios con `churned = 1`, buscando un F1-Score y Recall iguales o superiores a 0.90.

## 📊 Variables del modelo

**Variable objetivo:** `churned`

- `0` — El usuario permanece suscrito.
- `1` — El usuario canceló su suscripción.

**Variables predictoras seleccionadas:**

`age`, `watch_hours`, `last_login_days`, `subscription_type`, `device`, `payment_method`, `number_of_profiles`, `avg_watch_time_per_day`.

Estas características permiten estudiar los hábitos de consumo, la actividad y las condiciones de suscripción de los usuarios.

## 🛠️ Tecnologías utilizadas

- **Python:** Análisis y procesamiento de datos.
- **Pandas y NumPy:** Manipulación y preparación del dataset.
- **Matplotlib, Seaborn y Plotly:** Visualización de datos.
- **Scikit-learn:** Entrenamiento y evaluación de modelos predictivos.
- **SAS:** Análisis estadístico y trabajo con modelos de clasificación.
- **Streamlit:** Desarrollo de dashboards interactivos.
- **Jupyter Notebook:** Desarrollo y documentación del análisis.

## 🤖 Modelos de Machine Learning

Se compararon seis algoritmos de clasificación:

**Naive Bayes · KNN · Decision Tree · Random Forest · Neural Networks (MLP) · SVM**

En la evaluación original, **Random Forest** presentó el mejor F1-Score:

| Métrica | Resultado |
|---|---|
| Accuracy | 94.20% |
| Precision | 92.22% |
| Recall | 96.62% |
| F1-Score | 94.37% |
| ROC-AUC | 99.21% |

*Estos resultados corresponden a la comparación experimental realizada; aún requieren validación independiente para confirmar su capacidad de predicción futura.*

## 📂 Documentación y evidencias

El repositorio contempla los análisis realizados en Python y SAS, junto con la documentación de las hipótesis, visualizaciones, resultados de los modelos y capturas de los procedimientos utilizados.

Las evidencias de SAS se pueden organizar en carpetas para conservar las gráficas y resultados sin sobrecargar el README principal.

---

**Dataset:** Netflix Customer Churn and Engagement — Kaggle.

**Proyecto académico:** Tecnológico de Monterrey | Analítica de Datos e Inteligencia Artificial — Equipo 1.
