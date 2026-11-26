[CI contraste verde de la base revisada · 23585af](https://github.com/RICK1130042/Netflix_clasificadores/actions/runs/45721406065)

# Entregable 3 — Netflix Customer Churn Analytics

## 1. Gráficas y preguntas de negocio

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

## 2. Paleta y contraste WCAG 2.2 AA

La fuente única de los colores es `palette.py`. `ui.py` y `app.py` importan sus valores; el CI verifica los pares declarados y la coherencia de `.streamlit/config.toml` mediante `tests/test_contrast.py`. Ejecutar `python palette.py` para obtener todas las razones.

| Color / uso | Hex | Fondo #111318 | Tarjeta #1B1F27 |
|---|---|---:|---:|
| Texto principal | #F5F6FA | 17.21:1 | 15.29:1 |
| Texto secundario | #B9BEC8 | 9.97:1 | 8.86:1 |
| Acento / riesgo | #FF6B6B | 6.70:1 | 5.95:1 |
| Verde / cumplimiento | #46D369 | 9.54:1 | 8.48:1 |
| Ámbar / aviso | #F5A524 | 9.10:1 | 8.09:1 |
| Azul / gráficos | #56B4E9 | 8.05:1 | 7.15:1 |
| Gris / gráficos | #9AA0AA | 7.06:1 | 6.28:1 |
| Borde funcional | #788394 | 4.84:1 | 4.30:1 |

Umbrales AA: 4.5:1 para texto normal y 3:1 para texto grande y componentes gráficos relevantes. El texto y los símbolos/etiquetas identifican las categorías independientemente del color; se usan trazos diferentes para curvas, facetas y nombres de categorías. El CI prueba los tokens, no el DOM renderizado ni todos los estados internos de Streamlit/Plotly. Verificar también el tema, hover, selección, foco, barras, leyendas y simulaciones de protanopia, deuteranopia y tritanopia en la app publicada antes de afirmar cumplimiento integral.

Fuente normativa: [WCAG 2.2](https://www.w3.org/TR/WCAG22/), criterios 1.4.1, 1.4.3 y 1.4.11.

## 3. Jerarquía visual e Investigación 1

El resumen ejecutivo usa KPI de abandono y tamaño de población, luego evidencia gráfica, interpretaciones y notas metodológicas. Se prioriza el mensaje con tamaño, peso, posición y espaciado en lugar de solo color. Los detalles técnicos se reservan para la sección de modelos.

**PENDIENTE DEL EQUIPO:** citar el autor, título, año, URL y hallazgo concreto de la Investigación 1; explicar qué decisión visual se deriva de ese hallazgo. No sustituir por una referencia inventada. Verificar la posición de los KPI en la versión desplegada.

## 4. Estado vacío y `st.stop()`

En `app.py`, buscar `No seleccionaste ningún segmento` (selector de clúster), `No hay clientes con este filtro` (planes) y `No hay clientes con los filtros seleccionados. Ajusta la audiencia` (CRM). Cada caso muestra un mensaje y llama a `st.stop()` antes de calcular métricas/figuras o exportar una audiencia vacía. `Todos` es una elección explícita, no ausencia de selección.

Prueba pendiente de ejecutar en la aplicación: quitar todos los planes, deseleccionar el clúster, quitar las campañas y comprobar el mensaje y la ausencia de resultados posteriores.

## 5. Controles y decisiones de negocio

| Control | Decisión | Evidencia visible |
|---|---|---|
| Planes | Priorizar qué tipos de suscripción requieren análisis de retención | Planes activos y número de registros tras aplicar ambos filtros |
| Clúster | Adaptar la estrategia a un perfil de comportamiento | Segmento activo, nombre del perfil y conteo |

Se mantienen además la agrupación por variable y los controles de diseño de audiencia CRM. En el resumen, las métricas de los modelos permanecen globales; no se deben interpretar como métricas por clúster.

## 6. Matriz de acceso por rol y limitaciones

| Contenido | Gerencia (Laura) | Analista (demostración de CRM y modelado) |
|---|---|---|
| Resumen y exploración agregada | Sí | Sí |
| Panel gerencial de campañas | Sí | Sí |
| Comparar modelos y predicción individual | No | Sí |
| CRM, calidad y asignación A/B | No | Sí |
| Exportación de registros individuales | No | Sí |

`auth.py` valida una cuenta ficticia con credenciales de `st.secrets` y determina el rol a partir de la configuración del host, no de una selección del visitante. La navegación y las secciones protegidas se comprueban en el servidor antes de mostrar los contenidos. Cerrar sesión limpia `st.session_state`. Las credenciales no se incluyen en el repositorio.

Limitación: autenticación académica de dos roles en una aplicación Streamlit; no es un IAM empresarial. No incorpora MFA, gestión de contraseñas, limitación persistente de intentos, auditoría centralizada ni autorización a nivel de almacenamiento. El repositorio público y sus archivos no se protegen mediante el login de Streamlit. No deben contener datos personales reales; antes de entregar verificar procedencia, privacidad y acceso directo a CSV. Usar cuentas ficticias. Para producción, integrar proveedor de identidad, almacenamiento privado, controles de acceso sobre datos y auditoría.

**PENDIENTE:** contrastar esta matriz con las personas y permisos de la propuesta de octubre y la Investigación 3. Registrar prueba de inicio/cierre de sesión y cambio de rol en Cloud. Añadir evidencia de la Investigación 3 y completar `prompts/B3.md` según las instrucciones originales del curso.

## Auditoría posterior

Véase [auditoría de entrega](auditoria-entrega.md) y [evidencia CI](evidencia-ci.md). La primera liga acredita la base; las corridas de la revisión se identifican por su propio SHA. Los pendientes de investigación y autoría no se consideran cerrados.
