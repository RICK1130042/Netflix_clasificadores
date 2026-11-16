

# Guía breve para asistentes de IA — Netflix churn

## Propósito
Dashboard académico para explorar abandono y orientar priorización de segmentos. La fuente es `netflix_customer_churn.csv`; el archivo original tiene 5,000 filas. La app principal es `app.py` (Streamlit).

## Regla innegociable
- Nunca convertir valores numéricos inválidos en cero ni borrar filas en silencio.
- No mostrar KPIs, gráficas ni predicciones si falla `validacion_datos.py`.
- Mantener `churned` binario (`0` permanece, `1` cancela); no entrenar con `customer_id`.
- No mezclar métricas de diferentes experimentos: notebook original, comparativas posteriores, XGBoost y SAS deben citar su procedencia.
- Separar **DATO · fuente**, **CALCULADO · operación** y **SIMULADO · escenario**; la simulación no demuestra impacto de una campaña.
- No asumir causalidad ni predicción temporal futura. Revisar cuándo se midieron las variables de actividad.

## Comandos de verificación
- `python scripts/validar_contrato.py --csv netflix_customer_churn.csv` → salida `CONTRATO VÁLIDO` y código 0.
- `python -m unittest discover -s tests -v` → todas las pruebas `OK`.
- `streamlit run app.py` → indicador de contrato visible antes de usar datos.

## Forma de trabajo
Cambiar el código en una rama, ejecutar verificaciones, abrir PR y revisar resultados de GitHub Actions antes de incorporar cambios. El equipo decide reglas de negocio y ausencias; un asistente puede proponerlas, no aprobarlas por el equipo.
