
# Netflix | Churn Analytics · Streamlit

Dashboard construido a partir del notebook `netflix_churned.html`.

## Qué incluye

1. **Resumen ejecutivo**: 5,000 clientes, 50.3 % de abandono, objetivos del proyecto y resultado de la hipótesis.
2. **Explorar clientes**: filtros por plan, tasas de abandono por segmento, boxplot y mapa de correlaciones de Spearman (requiere CSV).
3. **Comparar modelos**: tablas y resultados reales del notebook; F1 en validación cruzada; escenarios con y sin variables; importancia por permutación; matrices de confusión, curvas ROC/PR y árbol (últimos cuatro requieren CSV).
4. **Predecir abandono**: simulador con Gradient Boosting; usa el `.joblib` exportado o reentrena con el CSV (requiere uno de ambos).

El dashboard **no incluye XGBoost** porque el notebook adjunto compara Regresión Logística, Árbol de Decisión, Random Forest y Gradient Boosting de scikit-learn.

## Archivos de la carpeta

```text
Netflix_clasificadores/
├── app.py
├── requirements_streamlit.txt
├── .streamlit/config.toml                # Tema oscuro de Netflix
├── netflix_customer_churn.csv           # Dataset original: colócalo aquí
├── modelo_churn_netflix.joblib           # Opcional: exportado desde el notebook
└── netflix_churned.ipynb                 # Tu notebook; opcional
```

El código presenta sin CSV las métricas publicadas en el notebook y avisa si faltan archivos para las funciones interactivas. No inventa predicciones o gráficas de clientes.

## Cómo ejecutarlo (macOS)

Desde Terminal, entra a la carpeta donde se encuentre `app.py`:

```bash
cd /ruta/a/Netflix_clasificadores
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_streamlit.txt
streamlit run app.py
```

Si ya utilizas `.venv` y tienes instaladas las dependencias, no hace falta crear otro entorno. Streamlit abrirá la página en `http://localhost:8501`.

## De dónde provienen las cifras

- Notebook original: resultados de comparación en validación cruzada y conjunto de prueba, importancia por permutación y conclusiones del EDA.
- CSV original: exploración y matrices/curvas calculadas a partir de los 5,000 registros.
- Modelo `.joblib`: predicciones individuales, cuando está disponible.

La selección de **Gradient Boosting** se realizó usando el F1 en validación cruzada (93.92 %); **Random Forest** tiene el F1 más alto en prueba (93.51 %) pero no se usa el conjunto de prueba para elegir la configuración.

> Importante: el promedio `avg_watch_time_per_day` se calcula en el simulador a partir de `watch_hours / (last_login_days + 1)`, siguiendo la relación observada en el notebook. Si cambiaste la fórmula en el CSV, actualízala también en `app.py`.

## Nota de compatibilidad

Los modelos `.joblib` de scikit-learn deben cargarse en un entorno con versiones compatibles (preferentemente las mismas del entrenamiento). Abre solo archivos `.joblib` que tú hayas generado.
