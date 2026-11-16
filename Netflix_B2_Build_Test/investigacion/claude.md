---
noteId: "f10c0162e15d11f18255d32709769dd3"
tags: []

---

# Investigación — El ciclo de desarrollo con asistentes de IA

**Entregable 2 · Plan, Design, Build, Test, Deploy y Maintain**  
**Proyecto:** Netflix — detección de patrones asociados con el abandono y apoyo a decisiones de retención.  
**Referencia principal:** Anthropic, *The AI-native SDLC playbook*, Claude Academy, lecciones 1 (introducción), 8 (*Give Claude a feedback loop*), 9 (*Continuous evals in CI*) y 5 (*The CLAUDE.md*). Consulta: 15 de noviembre de 2026.

## Resumen de investigación (extensión aproximada: dos cuartillas)

### 1. ¿En cuáles etapas ya usamos el asistente y en cuáles todavía no?

El ciclo descrito por Anthropic tiene seis etapas. En **Plan** se aclara qué problema se resolverá y para quién. En **Design** se convierte ese propósito en requerimientos y una estructura de pantallas. En **Build** se escribe el código. En **Test** se comprueba su funcionamiento. En **Deploy** se publica bajo controles de revisión. En **Maintain** se vigila su utilidad y su calidad después de publicarlo.

En el proyecto, **Plan y Design ya tienen evidencia de uso de IA**. Se discutió la hipótesis de clasificación de `churned = 1`, los criterios F1 y Recall de 0.90, las variables de actividad y las decisiones del usuario de negocio. Hubo solicitudes al asistente para mejorar un *wireframe*, explicar los indicadores y orientar qué segmentos visualizar. La propuesta de dashboard pasó de una vista general a una herramienta que sirve a quien decide una campaña. Este trabajo evidencia diseño, no solo programación.

En **Build** es donde más se ha usado. Entre los prompts registrados en nuestras conversaciones aparecen la petición de construir un dashboard con Streamlit a partir del notebook `netflix_churned`, la solicitud de mejorar `app.py` agregando un mapa de calor y un árbol, el entrenamiento y comparación de clasificadores y la pregunta por un `NameError` de `predicciones_test`. También hubo asistencia para usar GitHub desde VS Code y distinguir `pull`, `commit` y `sync`. El repositorio contiene código verificable: `app.py`, `scripts/data_pipeline.py`, `scripts/reporte_calidad.py`, notebooks y archivos de evaluación. Estas piezas permiten identificar exactamente dónde quedó el trabajo delegado.

**Test estaba incompleto como práctica automatizada.** Sí se analizaban métricas como Accuracy, Recall, F1, ROC-AUC y matrices de confusión; también había un reporte de calidad descriptivo que registraba nulos, duplicados y valores extremos. Pero **evaluar un clasificador no es lo mismo que verificar automáticamente el contrato de datos de la app**. Antes de este entregable, el programa podía recibir una columna de dinero como texto y mostrar cifras erróneas sin una barrera central que detuviera toda la interfaz. Este es el vacío que buscamos cerrar en B2 mediante tipos comprobados, pruebas negativas y un workflow.

En **Deploy** ya existe una publicación de dashboard en una URL compartida según el entregable anterior y un repositorio remoto. Por tanto, sí se ha trabajado en publicar y sincronizar, pero **no hay evidencia suficiente de que ya hubiera una revisión de PR o un check obligatorio de contrato previo al merge**. Tener una URL no demuestra ese control. Por último, en **Maintain** todavía no contamos con una serie de mediciones operativas, registro de errores en producción o evidencias del efecto de campañas reales; la presentación reconoce que el resultado es observacional y recomienda monitorear *drift*, calidad y decisiones antes de activar retención.

### 2. ¿Qué práctica del curso podemos adoptar esta semana?

La práctica más útil de la lección *Give Claude a feedback loop* es **darle al asistente un comando con el que pueda comprobar lo que construyó**. Anthropic propone que los tests se ejecuten antes de declarar una tarea terminada y que los fallos no se oculten cambiando o suprimiendo pruebas. Esto cambia la conversación: ya no basta decirle “haz un dashboard”; también debe pasar un criterio medible. En el proyecto adoptamos `python scripts/validar_contrato.py --csv netflix_customer_churn.csv` y `python -m unittest discover -s tests -v`. El primer comando termina con código 1 si el archivo no cumple el esquema; el segundo comprueba casos buenos y errores esperados. Un tercero, GitHub Actions, repite esos controles en `push` y `pull_request`.

La lección *Continuous evals in CI* propone evaluar nuevamente las configuraciones del agente cuando cambian sus instrucciones, habilidades o herramientas. Para nuestro tamaño de proyecto no es necesario copiar de inmediato un sistema corporativo de agentes con API y presupuestos propios; la adaptación responsable es **empezar por verificaciones deterministas de datos y pruebas regresivas**. Esas reglas no dependen de que el asistente recuerde una instrucción. Si alguien introduce `8.99 ERROR` en `monthly_fee`, el workflow debe fallar aunque el autor —humano o IA— crea que terminó. Las decisiones de negocio, en cambio, pueden necesitar revisión humana y admiten excepciones documentadas.

La regla orientadora del curso para **Build** es conservar las convenciones en un `CLAUDE.md` del repositorio. Ese archivo recuerda que no debemos inventar cifras, ocultar nulos, mezclar muestras de SAS con las del notebook ni afirmar causalidad. Es útil para orientar al asistente, pero **no sustituye una prueba automatizada**. Ambas capas se complementan: las instrucciones expresan la intención y el pipeline impide que se publique un dato inválido.

### 3. ¿Qué esperábamos encontrar y no estaba?

Esperábamos encontrar una receta exacta para corregir un CSV en Pandas, decidir qué hacer con cada `N/A` o generar automáticamente un dashboard de Streamlit que distinga datos reales y simulados. **El curso no ofrece ese tipo de contrato de datos específico para Netflix ni prescribe los umbrales de abandono**, porque está dirigido principalmente a equipos de ingeniería que organizan el ciclo completo y los controles de agentes. Tampoco promete que una instrucción en `CLAUDE.md` pueda hacer cumplir por sí sola una norma sin excepciones. Habla de *hooks*, tests, evaluaciones continuas y aprobaciones humanas: cada equipo debe decidir qué es verificable y qué exige juicio.

Esto es importante para nuestro proyecto. No sabemos solo por mirar un archivo si `monthly_fee` está expresado en dólares ni podemos inferir si un dato faltante significa “no hubo consumo”, “no aplica” o “no se capturó”. Son interpretaciones, no propiedades del tipo de dato. Tampoco podemos unificar una captura SAS de 2.5K usuarios con un dataset Python de 5,000 sin conocer filtros y procedencia. La contribución del equipo es **documentar** estas diferencias y hacer que el software se detenga cuando una regla confirmada se incumple.

**Conclusión:** en B2 avanzamos de usar IA para explicar y escribir código a usarla para ayudar a definir una prueba reproducible. La responsabilidad sobre el significado del dato y la aprobación de excepciones sigue siendo humana. En los siguientes entregables toca demostrar controles efectivos en publicación y después medir si el producto ayuda a tomar decisiones sin degradar la calidad de los datos.

## Tres cambios concretos en la forma de trabajar

| Cambio | ¿Dónde vive? | ¿Por qué ahí? |
|---|---|---|
| **1. Validar esquema y tipos del CSV en cada `push`/PR**, con pruebas que comprueban que el error detiene el proceso. | **DENTRO del pipeline**: `.github/workflows/calidad-datos.yml`, `validacion_datos.py` y `tests/`. | Es una regla sin excepciones silenciosas; el job debe fallar cuando el dato se rompe, independientemente de si alguien siguió un prompt. |
| **2. Registrar para qué sirve cada columna, sus nulos y el origen `DATO / CALCULADO / SIMULADO`.** | **FUERA del pipeline**: `contrato-datos.md`, revisión del equipo y texto visible en Streamlit. | El significado de un vacío o la decisión de negocio requiere criterio humano; el código implementa una regla después de que el equipo la aprueba. |
| **3. Pedir al asistente pruebas y evidencia antes de decir "listo".** | **FUERA del pipeline** como forma de trabajo en `CLAUDE.md` y `prompts/B2.md`; se comprueba ejecutando los comandos. | Mejora la calidad de la colaboración y permite auditar la intención; es orientación de trabajo, no una garantía automática. |

## Lecturas del curso

1. Introducción: https://academy.claude.com/courses/ai-native-sdlc-playbook/introduction
2. Test, ciclo de retroalimentación: https://academy.claude.com/courses/ai-native-sdlc-playbook/give-claude-a-feedback-loop
3. Test, evaluaciones continuas en CI: https://academy.claude.com/courses/ai-native-sdlc-playbook/continuous-evals-in-ci
4. Build, archivo `CLAUDE.md`: https://academy.claude.com/courses/ai-native-sdlc-playbook/claude-md

**Nota metodológica:** los ejemplos de prompts se identificaron en conversaciones disponibles del proyecto; para una auditoría literal de autor, fecha y texto, contrastar con el historial exportado de cada integrante. No se atribuyen a personas prompts que no constan en la evidencia.
