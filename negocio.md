# Objetivo de negocio — propuesta para validación del equipo

Decisión que habilita: priorizar segmentos y una audiencia candidata a ofertas de retención; la app no envía campañas ni acredita su efecto.
Persona: responsable de retención; confirmar con el equipo la persona de octubre y la frecuencia de revisión. La app distingue gerencia y analista.
Línea base descriptiva: el CSV versionado contiene 5,000 filas y 2,515 registros con churned=1 (50.3%). Recuento del archivo completo, sin filtros; no es una tasa mensual verificada ni una predicción futura.
Meta de negocio: pendiente de aprobación del equipo. El ejemplo docente de 70% de cancelaciones revisando 20% de la base no constituye una meta acordada ni un resultado medido.

La meta técnica de F1/Recall >=0.90 ya declarada en README/app.py no sustituye una meta de retención. Antes de activar una campaña deben confirmarse temporalidad, población activa, presupuesto, calibración y evaluación con grupo control.

Reproducir la línea base: `python -c "import pandas as pd; d=pd.read_csv('netflix_customer_churn.csv'); print(len(d),d.churned.sum(),d.churned.mean())"`.
