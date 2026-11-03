---
noteId: "aaaf90d0c5ae11f1a9d76d153424fc9c"
tags: []

---

# Dashboard Netflix Churn - versión ampliada

## 1. Copia los archivos

Coloca `app.py`, `requirements.txt` y `exportar_desde_notebook.py` en la **misma carpeta** que tu notebook y `netflix_customer_churn.csv`.

## 2. Corrige las celdas antiguas del notebook

No vuelvas a definir `mejor_k = 5` ni `modelo_knn_best` después de `GridSearchCV`. Para las curvas ROC y PR del notebook usa las puntuaciones de `puntajes_roc_pr` de cada modelo, nunca variables sobrantes de otro bucle.

## 3. Exporta después de ejecutar todas las celdas

Agrega **una celda al final** del notebook, después de los modelos y de las visualizaciones:

```python
from exportar_desde_notebook import exportar_dashboard
exportar_dashboard(globals())
```

Se generan `resultados_modelos.csv`, `evaluacion_modelos.csv`, `importancia_variables.csv`, `manifest_visualizaciones.csv`, `resumen_visualizaciones.json` y la carpeta `visualizaciones/`.

**Nota:** las regiones 2D y PCA son visualizaciones exploratorias de modelos auxiliares, no los modelos completos usados para calcular el rendimiento. El script solo vuelve a entrenar los dos KNN ilustrativos para alinear `k` con GridSearchCV; no reentrena el resto de modelos.

## 4. Instala y ejecuta desde la Terminal de tu Mac

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Si hay problemas de entorno, usa el `.venv` que ya creaste.

## Contenido

- Resumen: clientes, tasa de churn y distribución.
- Exploración: tasas por categoría y boxplots/histogramas.
- Clasificadores: comparación de seis modelos, matrices de confusión interactivas, curvas ROC y Precision-Recall.
- Visualizaciones: importancia de variables de árboles, árbol de decisión, regiones con dos características y PCA.

No se entrena ningún modelo dentro de Streamlit.
