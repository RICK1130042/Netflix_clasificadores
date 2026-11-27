[Dashboard en Streamlit Cloud](https://rick1130042-netflix-clasificadores-app-rzfx7h.streamlit.app/)

# Netflix · Customer Churn Analytics · Equipo 1

Proyecto académico de Analítica de Datos e Inteligencia Artificial del Tecnológico de Monterrey. Equipo 1: Gabriel Castro Rivas, Itzel Mejía Moreno, Juan Carlos Muñoz Flores, Kerstin Carmina Meister González, Paola Estefanía Alvarez Santoyo y Ricardo Villarreal Bazán.

## Situación problema

En un servicio de suscripción como Netflix, el abandono ocurre cuando un usuario cancela su plan. Si el equipo de retención tiene capacidad limitada de contacto, necesita saber qué segmentos concentran más cancelaciones y qué señales de uso pueden orientar una prueba de intervención. Este proyecto estudia la relación entre la cancelación y variables como las horas de visualización, los días desde el último acceso y el plan contratado. La hipótesis exploratoria es que una menor actividad y una inactividad más prolongada pueden estar asociadas con el abandono.

El tablero apoya dos decisiones. **Gerencia** puede comparar volúmenes y tasas por segmento para priorizar dónde investigar. **Analista** puede examinar modelos, explorar puntuaciones individuales y preparar una audiencia para una prueba de retención. La efectividad de una oferta se debe medir con un grupo de control y datos posteriores a la intervención. [Objetivo, línea base y meta propuesta](negocio.md).

## Datos y variables

La unidad de análisis es un usuario de la muestra académica. La fuente está identificada en el proyecto como *Netflix Customer Churn and Engagement*, de Kaggle. El CSV de origen contiene **5,000 registros y 14 variables**. El tablero usa **4,990 registros**, con **2,515 cancelaciones observadas (50.4%)**. Se documentó la exclusión de diez observaciones con un promedio diario superior a 24 horas. El CSV de origen se conserva para reproducir la preparación. Este porcentaje describe la muestra del proyecto. La ventana de observación no permite interpretarlo como una tasa mensual de la empresa. [Contrato de datos](contrato-datos.md) · [Reporte del preflight](datos/preflight.json).

| Tipo | Variables | Uso |
|---|---|---|
| Identificador | `customer_id` | Identificar registros y relacionar archivos. Se excluye del entrenamiento. |
| Objetivo binario | `churned` | `0` permanece suscrito; `1` canceló. |
| Numéricas de actividad | `watch_hours`, `last_login_days`, `avg_watch_time_per_day`, `number_of_profiles` | Medir intensidad de uso e inactividad. Las cuatro se usan en el escenario base del clasificador y se estandarizan para K-Means. |
| Categórica del servicio | `subscription_type` | Identificar el plan Basic, Standard o Premium. Se codifica para la clasificación y sirve para segmentar las vistas. |
| Contexto del usuario | `age` (entera) y `gender`, `region`, `device`, `payment_method`, `favorite_genre` (categóricas) | Describir perfiles y apoyar el análisis exploratorio, según la vista. |
| Cuota | `monthly_fee` (decimal) | Explorar el contexto económico. La moneda de la fuente está por confirmar. |

## Modelos y resultados

La versión final compara regresión logística, árbol de decisión, Random Forest, Gradient Boosting y XGBoost. Utiliza Python, pandas, scikit-learn, XGBoost, Plotly, Streamlit y notebooks de Jupyter. GitHub Actions verifica el contrato, la regresión, el contraste y la publicación de documentación.

| Familia | Resultado | Interpretación |
|---|---|---|
| Clasificación supervisada | XGBoost fue seleccionado por F1 medio de **0.9408** en validación cruzada de cinco particiones. En la prueba histórica obtuvo F1 **0.9329**, recall **0.9125** y ROC-AUC **0.9907**. | En esa prueba detectó alrededor de 91 de cada 100 cancelaciones. Son resultados de la cohorte histórica, separados del CSV activo. |
| Segmentación no supervisada | K-Means con **6 grupos**, silueta **0.2956** e inercia **6,595.21**, calculadas sobre las 4,990 observaciones validadas. | Los perfiles sirven para explorar diferencias de comportamiento. La silueta indica separación parcial y solapamiento entre grupos. |

Antes de usar los puntajes en una campaña real, hay que comprobar que las variables se midieron antes de la cancelación y validar el modelo en una cohorte posterior. [Métodos, fuentes y resultados reproducibles](investigacion/04-metricas.md).

## Entrega y evidencias

**Evaluación.** Abra el dashboard en ventana privada. La aplicación tiene cuentas de demostración para gerencia y analista. El equipo comparte esas credenciales de forma privada. La presentación final se entrega en Canvas.

**Documentación viva.** [Sitio de la entrega](https://rick1130042.github.io/Netflix_clasificadores/) · [Pipeline de publicación](https://github.com/RICK1130042/Netflix_clasificadores/actions/workflows/documentacion.yml).

| Artefacto | Dónde encontrarlo |
|---|---|
| App, dependencias y datos | [app.py](app.py), [requirements.txt](requirements.txt), [datos/](datos/) |
| Decisión, línea base y meta | [negocio.md](negocio.md) |
| Contratos | [Prompt](contrato-prompt.md) · [Datos](contrato-datos.md) |
| Gráficas, jerarquía y accesibilidad | [Fichas](graficas.md) · [Evidencia B3](docs/entregable-3.md) |
| Acceso y datos sensibles | [Matriz](acceso.md) · [Pruebas de roles](tests/test_roles.py) |
| Prompts | [B1](prompts/B1.md) · [B2](prompts/B2.md) · [B3](prompts/B3.md) · [B4](prompts/B4.md) · [Análisis por etapa](prompts/analisis-sdlc.md) |
| Investigaciones | [SDLC](investigacion/00-sdlc-genai.md) · [Jerarquía](investigacion/01-jerarquia.md) · [Contraste](investigacion/02-contraste.md) · [Acceso](investigacion/03-acceso.md) · [Métricas](investigacion/04-metricas.md) |
| Paleta y convenciones | [estilos.py](estilos.py) · [CLAUDE.md](CLAUDE.md) |
| Evolución del diseño | [Design thinking y capturas](docs/design-thinking.md) |
| Regresión y retroalimentación | [Corridas](docs/evidencia-ci.md) · [Observaciones de octubre](docs/observaciones-21-oct.md) · [Checklist de la rúbrica](docs/entrega-final.md) |

## Verificación local

```sh
python -m pip install -r requirements.txt
python scripts/preparar_entrega.py --check
python scripts/validar_contrato.py --csv datos/netflix_validado.csv
python -m unittest discover -s tests -v
streamlit run app.py
```

El proyecto usa secretos locales o de Streamlit Cloud para las cuentas. Las pruebas usan cuentas efímeras propias. Consulte el [contrato](contrato-datos.md) para reproducir la preparación y distinguir las métricas históricas de la evaluación sobre el CSV activo.
