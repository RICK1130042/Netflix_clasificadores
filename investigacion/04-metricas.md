# 04 · Métricas de desempeño de las dos familias

| Familia | Resultado reproducible | Qué implica para el producto |
|---|---|---|
| Supervisada, clasificador XGBoost elegido por CV | F1 medio 0.9408 en cinco folds. En prueba histórica: F1 0.9329, recall 0.9125, precisión 0.9543 y ROC-AUC 0.9907 | En esa prueba detectó alrededor de 91 de cada 100 cancelaciones. La precisión indica cuántas alertas eran cancelaciones con ese umbral. El AUC describe ordenamiento y no valida la calibración del score ni el impacto de una oferta. |
| No supervisada, K-Means con K=6 | Inercia 6,595.21 y silueta 0.2956 en 4,990 registros con cuatro variables estandarizadas | La silueta positiva sugiere cierta separación, con perfiles todavía solapados. Conviene usar los seis perfiles para explorar y priorizar, sin tratarlos como clases reales. La inercia solo es comparable con la misma escala y cohorte. |

Fuentes supervisadas: [CV](../comparacion_validacion_cruzada.csv) y [prueba](../comparacion_modelos_churn.csv). Corresponden a la evaluación histórica. Random Forest lidera F1 en esa tabla de prueba, mientras que la regla de selección usa CV. Los resultados de prueba no se usan para seleccionar el modelo. El ROC-AUC de K-Means + regresión logística que figura en la app es de la regresión, no de K-Means en solitario.

Reproducción de K-Means: `python scripts/metricas_segmentacion.py`. Véase la [documentación de scikit-learn sobre silueta e inercia](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient). Esta medición descriptiva usa la cohorte de entrega. No sustituye la evaluación temporal de retención.
