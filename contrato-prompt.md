# Netflix Customer Churn — Desarrollo del Dashboard

## 1. ¿Qué debe mostrar?

El dashboard permitirá a una analista de datos explorar el comportamiento de los clientes de Netflix e identificar patrones relacionados con el abandono de suscripciones.

- **Panorama del abandono:** Indicadores generales y comparación de modelos predictivos.
- **Patrones principales:** Comparación de horas de visualización y días desde el último acceso.
- **Explorar el comportamiento:** Filtros por tipo de suscripción y tasas de abandono por segmento.
- **Comportamiento y correlaciones:** Boxplots, matriz de correlación e interpretación de resultados.

Su propósito es facilitar la identificación de segmentos vulnerables y apoyar decisiones relacionadas con la retención de clientes.

---

## 2. El diseño de lo que debe arrojar

El dashboard presentará:

- **Total de clientes analizados:** Número de clientes.
- **Tasa de abandono:** Porcentaje (%).
- **F1-score y Recall:** Porcentaje (%).
- **Mediana de horas de visualización:** Horas.
- **Días desde el último acceso:** Días.
- **Clientes por segmento:** Número de clientes.
- **Tasa de abandono por suscripción:** Porcentaje (%).
- **Correlaciones estadísticas:** Escala de -1 a 1.

La información se organizará mediante:

- Tarjetas de indicadores.
- Gráficas de barras.
- Boxplots.
- Tablas comparativas.
- Mapas de calor.

### Diseño visual

El diseño utilizará:

- **Fondo:** Oscuro.
- **Tarjetas de indicadores:** Grises.
- **Gráficas:** Rojas.
- **Distribución:** Clara, organizada y profesional.

Todo siguiendo el mockup definido por el equipo.

---

## 3. El prompt

Actúa como un desarrollador experto en Python, Streamlit, análisis de datos y Machine Learning.

Necesito que desarrolles un dashboard interactivo en Streamlit para nuestro proyecto académico **Netflix Customer Churn**, utilizando el dataset de Kaggle que te proporcionaré.

El usuario principal será una analista de datos de Netflix, quien utilizará el dashboard para identificar patrones de abandono, detectar segmentos con mayor riesgo de cancelación y apoyar la toma de decisiones relacionadas con la retención de clientes.

Te proporcionaré nuestra presentación y las imágenes del mockup.

También quiero que hagas otra sección con dashboards para los otros usuarios de la presentación, donde se refleje su uso de Netflix o un resumen tipo **Netflix Wrapped**, incluyendo información como película favorita, serie favorita y otros patrones de consumo.

Respeta el diseño de la presentación utilizando fondo oscuro, tarjetas de indicadores, gráficas rojas y una distribución clara y profesional.

### Secciones del dashboard

#### 3.1. Panorama del abandono

Debe incluir:

- Total de clientes.
- Tasa de abandono.
- F1-score.
- Recall.
- Comparación de modelos predictivos:
  - Gradient Boosting.
  - Random Forest.
  - Árbol de Decisión.
  - Regresión Logística.

#### 3.2. Patrones principales

Debe incluir:

- Comparación de horas de visualización.
- Comparación de días desde el último acceso.
- Análisis entre clientes que permanecen y clientes que cancelan.

#### 3.3. Explorar el comportamiento

Debe incluir:

- Filtros por tipo de suscripción:
  - Basic.
  - Standard.
  - Premium.
- Indicadores dinámicos.
- Tasa de abandono por segmento.
- Tablas comparativas.

#### 3.4. Comportamiento y correlaciones

Debe incluir:

- Boxplot de horas de visualización.
- Matriz de correlación de Spearman.
- Interpretación de los principales hallazgos.

### Requisitos técnicos

- Utiliza **Python, Streamlit, Pandas, Plotly y Scikit-learn**.
- Todas las métricas y gráficas deben calcularse con el dataset real.
- Incluye filtros interactivos y descarga de datos en CSV.
- Agrega interpretaciones que expliquen los resultados y su relevancia para el negocio.
- Evita inventar datos o presentar correlaciones como causalidad.
- Respeta los resultados reales del análisis y la selección del modelo mediante validación cruzada.
- Incluye manejo de datos faltantes y errores.

### Entregables

Genera el código completo y organizado, incluyendo:

- `app.py`
- `requirements.txt`
- `README.md`

### Validaciones

Antes de programar, revisa el dataset y la presentación.

No olvides verificar que:

- El dashboard funcione correctamente.
- Los filtros actualicen las gráficas.
- Los indicadores coincidan con los datos reales.
- Los modelos predictivos presenten resultados válidos.
- Las visualizaciones respeten el diseño establecido.

---

## 4. ¿Qué devolvió la IA?

La IA generó una primera versión del dashboard utilizando **Streamlit**, aunque inicialmente presentó algunos problemas de funcionamiento y visualización.

Posteriormente, se consiguieron implementar las cuatro secciones solicitadas:

1. **Panorama del abandono:** Indicadores generales y comparación de modelos predictivos.
2. **Patrones principales:** Gráficas comparativas sobre el comportamiento de los clientes.
3. **Explorar el comportamiento:** Filtros interactivos para analizar los diferentes segmentos de clientes.
4. **Comportamiento y correlaciones:** Visualizaciones estadísticas e interpretación de los resultados.

La primera versión permitió establecer una base funcional para continuar con el desarrollo y los ajustes visuales del proyecto.

---

## 5. ¿Qué cambiamos de lo que devolvió?

Después de revisar la primera versión generada por la IA, se identificaron diferentes aspectos que necesitaban mejoras.

### Cambios realizados

- **Modificación de gráficas:** Se cambiaron algunas visualizaciones dentro de los dashboards de usuarios, debido a que no mostraban completamente la información o no contaban con una presentación visual adecuada.

- **Mejoras en el diseño:** Se realizaron ajustes para conseguir una interfaz más agradable, clara y organizada.

- **Separación de dashboards:** Inicialmente, los tres dashboards aparecían combinados dentro de una misma página o interfaz.

- **Organización de interfaces:** Se decidió separar los dashboards para que cada uno contara con su propio espacio, facilitando su navegación y comprensión.

Estos cambios permitieron mejorar la experiencia del usuario y presentar la información de manera más clara.

---

## 6. ¿Qué decidimos nosotros y no el modelo?

Como equipo, tomamos las decisiones principales relacionadas con el desarrollo y la finalidad del proyecto.

### Decisiones del equipo

- **Objetivo de negocio:** Identificar patrones de abandono de suscripciones y apoyar estrategias de retención de clientes.

- **Perfil del usuario:** Definir a la analista de datos de Netflix como uno de los usuarios principales del dashboard.

- **Métricas prioritarias:** Seleccionar los indicadores necesarios para evaluar el comportamiento de los clientes y el rendimiento de los modelos predictivos.

- **Estructura del dashboard:** Establecer las secciones, la organización de la información y las visualizaciones requeridas.

- **Preguntas de análisis:** Determinar qué preguntas debía responder el dashboard para generar información útil.

- **Interpretación de resultados:** Definir la importancia de los hallazgos obtenidos y su relación con el abandono de clientes.

- **Identificación de segmentos vulnerables:** Priorizar el análisis de los grupos de clientes con mayor tasa de cancelación.

- **Decisiones de negocio:** Establecer cómo los resultados podrían apoyar estrategias para mejorar la retención de suscriptores.

### Papel de la inteligencia artificial

La inteligencia artificial se utilizó como una herramienta de apoyo para:

- Generar la estructura inicial del código.
- Desarrollar componentes del dashboard.
- Implementar visualizaciones.
- Facilitar la integración de herramientas de análisis de datos.
- Apoyar la corrección de errores y las mejoras de la interfaz.

Sin embargo, **las decisiones sobre el producto, la definición de los objetivos, la selección de métricas, la interpretación de resultados y la validación final del dashboard corresponden al equipo**.

La IA funcionó como una herramienta de asistencia durante el desarrollo, mientras que el equipo mantuvo el control sobre las decisiones técnicas, analíticas y de negocio.
