# Fichas de las 20 gráficas

## Preguntas y decisiones

| Página | Gráfica | Pregunta que responde |
|---|---|---|
| Resumen | F1 en CV | ¿Qué modelo se debe seleccionar por F1 en validación cruzada? |
| Explorar | Abandono por segmento | ¿Qué segmento presenta mayor tasa de abandono y merece atención prioritaria? |
| Explorar | Horas vistas (cajas) | ¿Qué diferencias de horas vistas existen entre quienes permanecen y quienes cancelan? |
| Explorar | Spearman con abandono | ¿Qué variables se asocian más con el abandono y merecen seguimiento? |
| Modelos | Comparación en prueba | ¿Cómo se comparan los modelos en prueba según la métrica seleccionada? |
| Modelos | Codo | ¿A partir de qué número de clústeres disminuye la mejora de la compactación de grupos? |
| Modelos | ROC de K-Means + logística | ¿Cómo cambia la discriminación del abandono al usar K = 6 frente a otro K? |
| Modelos | AUC por K | ¿Qué cantidad de clústeres merece validación adicional por su AUC exploratorio? |
| Modelos | Tamaño de clústeres | ¿Qué tamaño y tasa de abandono tiene cada grupo para orientar la capacidad de retención? |
| Modelos | Importancia de variables | ¿Qué variables aportan más al desempeño del modelo y conviene vigilar? |
| Modelos | Matriz de confusión (barras) | ¿Cuántos falsos negativos y falsos positivos produce el modelo seleccionado? |
| Modelos | ROC | ¿Qué equilibrio entre detección y falsas alarmas ofrece el modelo al variar el umbral? |
| Modelos | Precision-Recall | ¿Qué equilibrio entre precisión y recall ofrece el modelo al variar el umbral? |
| Campañas | Prioridades | ¿Cuántos clientes requieren cada nivel de atención para dimensionar la campaña? |
| Campañas | Prioridad alta por plan | ¿En qué plan hay más clientes de prioridad alta para asignar recursos? |
| Campañas | Distribución de scores por prioridad | ¿Cómo se distribuyen los scores por prioridad para revisar la cartera? |
| Predicción | Indicador de riesgo | ¿Qué riesgo estima el modelo para el perfil y qué prioridad corresponde? |
| Predicción | SHAP, si el modelo lo permite | ¿Qué variables aumentan o reducen la predicción de abandono? |
| Predicción | Simulación de días sin acceso | ¿Cómo cambia el riesgo estimado si varían los días sin acceso? |
| Predicción | Simulación de horas vistas | ¿Cómo cambia el riesgo estimado si varían las horas vistas? |

Se propone retirar el árbol ilustrativo porque no necesariamente representa el clasificador seleccionado. La matriz de confusión conserva sus cuatro conteos y la gráfica de Spearman se concentra en la asociación con churn; las asociaciones no prueban causalidad.

## Lectura

Cada gráfica debe responder la pregunta de su fila. El color apoya etiquetas y títulos; los filtros y estados vacíos se describen en [la ficha B3](docs/entregable-3.md). Las cifras de modelos exportadas son históricas y las vistas recalculadas usan el CSV activo.
