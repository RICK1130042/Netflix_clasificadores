# Contrato de datos del tablero

Las reglas de tipos, unidades, ausencias y columnas derivadas se mantienen en el [contrato B2 de 14 columnas](Netflix_B2_Build_Test/contrato-datos.md). La fuente original se conserva en `netflix_customer_churn.csv` y la versión que consume Streamlit en [datos/](datos/).

El archivo original contiene 5,000 filas. Diez tienen `avg_watch_time_per_day > 24`. La regla documentada interpreta esa columna como horas diarias. Se excluyen las diez filas del CSV de entrega y se conserva la fuente original íntegra para revisión. La decisión de preparación es explícita y reversible: no se recortan valores ni se imputan mediciones. Si el proveedor confirma una unidad distinta, se debe revisar el contrato y regenerar el CSV. Permanecen 4,990 filas y 2,515 cancelaciones. La bitácora [preflight.json](datos/preflight.json) registra los conteos y el hash.

Desde la raíz:

```sh
python scripts/preparar_entrega.py --check
python scripts/validar_contrato.py --csv datos/netflix_validado.csv
```

La validación del CSV de entrega exige las mismas reglas que el original. La validación estricta del original mantiene el hallazgo histórico. El reporte del preflight no acredita la procedencia o temporalidad de cada predicción. Las métricas y modelos ya exportados corresponden a la cohorte histórica y se identifican como tales.

Antes de publicar un CSV, quien prepara la revisión adjunta fuente, hash y diff del reporte al PR. Quien aprueba coteja el preflight, las unidades, el recuento y la privacidad. La asignación nominal permanente la confirma el equipo.
