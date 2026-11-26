

# Contrato de datos del dashboard Netflix — Build y Test (B2)

**Versión:** 1.0 · **Fuente operativa:** `netflix_customer_churn.csv` · **Tablero:** `app.py` (Streamlit)

## Qué conjunto de datos respalda cada cifra

El CSV original disponible en el repositorio contiene **5,000 usuarios y 14 columnas**. El archivo preparado `data/prepared/netflix_limpio.csv` tiene 15 columnas porque agrega `segmento_inactividad`. El objetivo `churned` significa **0 = permanece** y **1 = cancela**. El resumen registrado del pipeline indica 2,515 abandonos (50.3%) y 0 nulos en la versión original.

La **presentación "Netflix Abandono.pdf"** contiene además resultados de Random Forest y capturas de **SAS**. Es importante NO consolidar sin evidencia sus cifras con la app de Streamlit: el material muestra, entre otras cosas, **2.5K** usuarios en una pantalla de SAS y una visualización de **505/495** en otra. Pueden obedecer a filtros, muestras o análisis diferentes; **no se ha demostrado que correspondan a las mismas 5,000 filas**. La app principal del repositorio utiliza el CSV de 5,000 y presenta sus experimentos del notebook por separado.

### 1. Contrato de las 14 columnas del CSV original

| Nombre | Tipo obligatorio tras validar | Unidad | Origen de la cifra | Significado de celda vacía |
|---|---|---|---|---|
| `customer_id` | Texto/identificador único | ID, sin unidad | **DATO** · CSV de Kaggle | **Error bloqueante:** no se puede identificar el registro |
| `age` | Entero | Años | **DATO** · CSV original | **Error bloqueante:** edad no registrada; **no** equivale a 0 |
| `gender` | Texto/categoría | Categoría | **DATO** · CSV original | Ausente: género no registrado; desconocido, **no** otro género |
| `subscription_type` | Texto/categoría | Basic / Standard / Premium en la muestra original | **DATO** · CSV original | **Error bloqueante:** falta el plan para segmentar |
| `watch_hours` | Decimal | Horas (ventana temporal sin confirmar) | **DATO** · CSV original | **Error bloqueante:** horas no registradas; **no** equivale a 0 horas |
| `last_login_days` | Entero | Días desde último acceso | **DATO** · CSV original | **Error bloqueante:** no se conoce la inactividad; **0** sí significa 0 días |
| `region` | Texto/categoría | Región | **DATO** · CSV original | Ausente: región no registrada; no crear región ficticia |
| `device` | Texto/categoría | Dispositivo | **DATO** · CSV original | Ausente: dispositivo no registrado; no equivale a "ninguno" |
| `monthly_fee` | Decimal | Unidad monetaria del archivo; **moneda no verificada** | **DATO** · CSV original | **Error bloqueante:** cuota desconocida; 0 no reemplaza un faltante |
| `churned` | Entero binario `0/1` | Categoría binaria | **DATO** · CSV original | **Error bloqueante:** no se sabe si canceló; no asignar 0 |
| `payment_method` | Texto/categoría | Método | **DATO** · CSV original | Ausente: método no registrado, no implica ausencia de pago |
| `number_of_profiles` | Entero | Cantidad de perfiles | **DATO** · CSV original | **Error bloqueante:** número desconocido; no asignar 0 |
| `avg_watch_time_per_day` | Decimal | Horas/día | **DATO** · CSV original | **Error bloqueante:** promedio desconocido; no imputarlo por intuición |
| `favorite_genre` | Texto/categoría | Género de contenido | **DATO** · CSV original | Ausente: género favorito no disponible; **no** asumir que no aplica |

> **Responsabilidad humana:** esta política de ausencias es una propuesta conservadora para el entregable, consistente con la versión del archivo (sin nulos). El equipo debe aprobar formalmente cualquier excepción futura. No debe cambiarse silenciosamente para que un test pase.

### 2. Columna creada en el pipeline de preparación

| Nombre | Tipo | Unidad | Origen de la cifra | Vacío |
|---|---|---|---|---|
| `segmento_inactividad` | Texto/categoría | Tramo 0–7, 8–30 o 31+ días | **CALCULADO** · `pd.cut(last_login_days)` en `scripts/data_pipeline.py` | Si no puede calcularse por ausencia de `last_login_days`, hay **error**. Un tramo que no concuerda con su valor fuente también bloquea. |

### 3. Columnas publicadas en las tablas y KPI del dashboard

Estas columnas se **visualizan con nombres legibles** en la nueva página `🧪 Calidad del dato`. Los nombres técnicos de la sección 1 siguen siendo la fuente de verdad.

| Nombre visible | Tipo | Unidad | Origen de la cifra | Vacío |
|---|---|---|---|---|
| Plan · DATO | Texto | Plan | Columna `subscription_type` del CSV | No permitido en esta tabla |
| Cuota mensual · DATO | Decimal real, no texto | Unidad monetaria sin confirmar | `monthly_fee` del CSV | Bloquea la tabla y el tablero |
| Horas vistas · DATO | Decimal real | Horas | `watch_hours` del CSV | Bloquea |
| Días sin acceso · DATO | Entero real | Días | `last_login_days` del CSV | Bloquea |
| Cancelación (0/1) · DATO | Entero real | 0/1 | `churned` del CSV | Bloquea |
| Fuente | Texto | Referencia | Nombre literal del CSV | No se permite vacío |
| Plan | Texto | Plan | Agrupación por `subscription_type` | No se permite vacío |
| Clientes · CALCULADO | Entero | Usuarios | Conteo de filas por plan | 0 significa cero usuarios; grupo sin filas no se genera |
| Abandonos · CALCULADO | Entero | Usuarios | Suma de `churned` por plan | 0 significa cero cancelaciones del grupo |
| Abandono (%) · CALCULADO | Decimal real | % | `100 * Abandonos / Clientes` | No se calcula si el denominador es 0 |
| Origen / fórmula | Texto | Fuente/fórmula | Explicación del cálculo | No se permite vacío |
| Estado, Columna, Tipo esperado, Tipo encontrado, Nulos | Texto/entero según campo | Comprobación | **CALCULADO** · `validacion_datos.py` | Error en el campo o en la validación; no se oculta |

En las otras secciones de la app, los KPI `Clientes analizados` (conteo) y `Tasa de abandono` (media de `churned` × 100) son **CALCULADOS** cuando el CSV existe. Las cifras de desempeño incrustadas del notebook son **RESULTADOS REPORTADOS**, no valores recalculados por el tablero ni resultados SAS actuales.

### 4. Tabla comparativa de modelos y simulador

| Nombre | Tipo | Unidad | Origen | Vacío |
|---|---|---|---|---|
| Modelo | Texto | Nombre | **REPORTADO** · notebook / `comparacion_modelos_churn.csv`, según la tabla | Modelo sin nombre: error |
| Accuracy | Decimal | Proporción 0–1 | **REPORTADO** · evaluación de prueba del experimento correspondiente | No disponible: mostrar N/D, nunca 0 |
| Precision | Decimal | Proporción 0–1 | **REPORTADO** · evaluación de prueba | No disponible: N/D |
| Recall | Decimal | Proporción 0–1 | **REPORTADO** · evaluación de prueba | No disponible: N/D |
| F1 / F1-Score | Decimal | Proporción 0–1 | **REPORTADO** · evaluación de prueba o media de validación cruzada, etiquetar cuál | No disponible: N/D |
| ROC-AUC | Decimal | Proporción 0–1 | **REPORTADO** · evaluación de prueba | No disponible: N/D |
| Escenario | Texto | Configuración | **REPORTADO** · experimento del notebook | No disponible: no comparar |
| F1 CV | Decimal | Proporción 0–1 | **REPORTADO** · validación cruzada | No disponible: N/D |
| Variable | Texto | Predictor | **REPORTADO** · importancia por permutación | Sin variable: error |
| Disminución del F1 | Decimal (puede ser negativo) | Puntos de F1 | **REPORTADO** · permutación del notebook | No disponible: N/D |
| Probabilidad estimada · SIMULADA | Decimal | Proporción / % | **SIMULADO** · perfil hipotético + modelo académico | Sin modelo o perfil: no mostrar predicción |
| Promedio diario calculado | Decimal | Horas/día | **CALCULADO SOBRE PERFIL SIMULADO** · `watch_hours / (last_login_days + 1)` | Entradas incompletas: no simular |

**Importante:** no declarar que la selección de Random Forest en la presentación y la selección de Gradient Boosting en la app son el mismo experimento. Las versiones usaron criterios y configuraciones distintos. Si se publica XGBoost, citar por separado `outputs/xgboost/metricas_xgboost.json` y sus predicciones de prueba, sin sustituir automáticamente métricas previas.

### 5. Ausencias, ceros y errores: decisiones explícitas

| Valor/tipo observado o posible | Interpretación propuesta | Transformación / conducta |
|---|---|---|
| Celda vacía, `NA`, `N/A`, `null`, `None`, `Sin dato`, `--`, `?` | **No hay dato documentado**. No es automáticamente cero ni "no aplica". | Se convierten en `pd.NA`. En campos críticos: bloquea. En categorías opcionales: queda nulo y se identifica. |
| `0` | **Dato válido** cuando la columna lo permite (p. ej. `churned=0`, `last_login_days=0`). | Se conserva como cero numérico; nunca se interpreta como celda vacía. |
| `ERROR`, `ERR`, `#N/A`, `#VALUE!`, `#DIV/0!`, `Inválido` | **Error explícito del archivo o de la hoja**. | Bloquea la visualización: no se elimina ni imputa. |
| "No aplica" | **Ausencia por no aplicabilidad**, distinta de desconocido. | No se ha confirmado que sea un código usado por este CSV. No se transforma de forma automática: primero se necesita la definición del proveedor. |
| `8.99 ERROR` en `monthly_fee` | Texto inválido en columna numérica. | Error bloqueante, evita ordenamiento lexicográfico y sumas incorrectas. |

Los CSV del repositorio se leen con tipos explícitos; el control exige numéricos verdaderos (`int64`/`float64`) y falla si hay valores incompatibles. En Streamlit, las tablas incluyen `column_order` y `hide_index=True`, y la columna monetaria utiliza formato numérico **sin cambiar su dtype**.

### 6. Reglas bloqueantes y reglas sujetas a criterio

**Bloqueantes en código/pipeline:** existencia de las 14 columnas, tipo numérico, `churned ∈ {0,1}`, cuota positiva, días/horas no negativos, perfiles enteros válidos, `customer_id` no duplicado, marcadores `ERROR` prohibidos, ausencia crítica prohibida y cálculo derivado coherente.

**Sujetas a decisión humana:** si se permite un valor faltante en un campo de negocio, moneda definitiva del archivo, cuándo se midió la actividad, qué segmento priorizar, umbral para contactar clientes, reglas de SAS y si un modelo está listo para operar. **La ausencia de temporalidad validada impide afirmar que el modelo pronostica cancelaciones futuras.**

## Fuente y verificación

- `README.md`, `app.py`, `scripts/data_pipeline.py` y datos originales, repositorio `RICK1130042/Netflix_clasificadores` (consulta de `main`, 15-nov-2026).
- `Netflix Abandono.pdf`, pp. 2–10 (resultados Random Forest y capturas SAS).
- `python scripts/validar_contrato.py`: falla con salida 1 si se rompe este contrato.

## Auditoría ejecutable 26-nov-2026

La página de Calidad del dato descrita arriba no está integrada en app.py raíz. Los comandos se ejecutan ahora desde la raíz con validacion_datos.py y scripts/validar_contrato.py; no existe validar_exportaciones.py. El contrato rechaza 10 valores avg_watch_time_per_day superiores a 24, tanto en fuente como preparado. No se alteran datos ni límites para obtener verde. Detalles y responsabilidades pendientes: ../docs/auditoria-entrega.md y ../docs/observaciones-21-oct.md.
