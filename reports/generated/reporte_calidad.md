# Calidad de datos — Netflix

- Filas: **5000**
- Columnas: **15**
- Duplicados exactos: **0**
- Duplicados de customer_id: **0**
- Tasa descriptiva de churn: **50.30%**

## Archivos
- `perfil_columnas_python.csv`: tipos y nulos por campo.
- `outliers_iqr.csv`: valores que ameritan revisión por regla IQR (no eliminación automática).
- `churn_por_segmento.csv`: distribución y tasas descriptivas por categoría.

## Precauciones de interpretación
- Un outlier estadístico no es necesariamente un error.
- Diferencias por grupo no establecen causalidad ni demuestran por sí solas sesgo del modelo.
- Para estudiar sesgo predictivo, medir también falsos negativos, recall y tamaños por grupo en test.
- Confirmar que las variables de actividad se observan antes del evento `churned`.
