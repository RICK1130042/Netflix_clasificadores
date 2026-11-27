# Checklist de la entrega final

Verificación de la entrega integrada en `main`, commit `1defeab`. **Completo** indica evidencia disponible y comprobada. **Parcial** indica material entregado con una limitación concreta. La puntuación corresponde al profesor.

## Rúbrica de 100 puntos

| Criterio | Pts | Estado y evidencia |
|---|---:|---|
| Métricas de desempeño de las dos familias, con número e interpretación | 24 | **Completo.** XGBoost reporta F1, recall, precisión y ROC-AUC en la cohorte histórica. K-Means reporta inercia y silueta en la cohorte validada. [Resultados, interpretación y reproducción](../investigacion/04-metricas.md). |
| Regresión de los binarios de los entregables 1, 2 y 3 al 24 de noviembre | 20 | **Parcial por fecha.** Se verificaron el formulario público, los roles en la app local, los tipos y el preflight, la tabla con orden e índice explícitos y el contraste. Los tres pipelines finales pasaron el 26 de noviembre. No se encontró una corrida completa fechada el 24. [Datos y 26 pruebas](https://github.com/RICK1130042/Netflix_clasificadores/actions/runs/45917717297) · [Contraste](https://github.com/RICK1130042/Netflix_clasificadores/actions/runs/45917717396) · [Fallos deliberados y reversiones](evidencia-ci.md). |
| Design thinking con evidencia real y cambio del mockup al producto | 20 | **Parcial.** Incluye especificación original versionada, propuesta individual, comparaciones del resumen y exploración, y cambios documentados. Falta el mockup original para comparar la transición directamente y evidencia de investigación con usuarios si existió. [Diseño y capturas](design-thinking.md). |
| Documentación viva completa, generada y publicada por el pipeline | 12 | **Completo.** GitHub Actions generó y publicó el índice y las fichas desde `main`. Se comprobó acceso público. [Sitio](https://rick1130042.github.io/Netflix_clasificadores/) · [Publicación](https://github.com/RICK1130042/Netflix_clasificadores/actions/runs/45917717355). |
| Análisis de prompts por etapa, con decisiones delegadas y humanas | 12 | **Parcial.** El análisis cubre Plan, Design, Build, Test, Deploy y Maintain. B1 no tiene historial literal recuperado; B2 contiene extractos y B3 carece de validación histórica completa. B4 registra la solicitud de integración y las decisiones observadas. [Análisis por etapa](../prompts/analisis-sdlc.md). |
| Defensa individual con preguntas aleatorias | 12 | **Pendiente de evaluación.** Cada integrante debe explicar la decisión de negocio, los roles, la preparación del CSV, las métricas de ambas familias y qué decisiones quedaron bajo criterio humano. |

## Los 14 artefactos solicitados

| # | Artefacto | Estado |
|---:|---|---|
| 1 | URL de la app | **Completo.** Primera línea del [README](../README.md). El enlace abre el formulario propio de la app desde un navegador sin sesión previa. |
| 2 | App, dependencias y datos | **Completo.** [App](../app.py), [dependencias](../requirements.txt), [datos](../datos/README.md). |
| 3 | Última presentación | **Pendiente en Canvas.** Confirmar que la versión entregada incorpora las observaciones finales del docente. |
| 4 | Objetivo de negocio | **Completo como propuesta.** [Decisión, línea base y meta](../negocio.md). El horizonte del piloto y su aprobación corresponden al equipo. |
| 5 | Contrato del prompt | **Completo.** [Especificación](../contrato-prompt.md). |
| 6 | Contrato de datos | **Completo.** [Contrato y preparación trazable](../contrato-datos.md). Falta designar a la persona responsable permanente de la revisión del CSV. |
| 7 | Fichas por gráfica | **Completo.** [20 fichas](../graficas.md) con pregunta de negocio. |
| 8 | Matriz de acceso y dato sensible | **Completo en el repo.** [Matriz y límites](../acceso.md). Falta demostrar ambas cuentas con credenciales del host en Cloud. |
| 9 | Registros B1 a B4 | **Parcial.** Están las cuatro rutas. El historial y las validaciones faltantes se declaran en [B1](../prompts/B1.md), [B2](../prompts/B2.md), [B3](../prompts/B3.md) y [B4](../prompts/B4.md). |
| 10 | Investigaciones | **Rutas completas.** [SDLC](../investigacion/00-sdlc-genai.md), [jerarquía](../investigacion/01-jerarquia.md), [contraste](../investigacion/02-contraste.md), [acceso](../investigacion/03-acceso.md) y [métricas](../investigacion/04-metricas.md). Las fichas 1 y 3 son revisiones realizadas para la entrega final. |
| 11 | Hoja de estilos | **Completo.** [estilos.py](../estilos.py) expone los tokens de la fuente única [palette.py](../palette.py). |
| 12 | Pipeline | **Completo.** [Workflows](../.github/workflows) de datos, contraste y documentación. |
| 13 | Convenciones para el asistente | **Completo.** [CLAUDE.md](../CLAUDE.md). |
| 14 | URL de documentación viva | **Completo.** En el [README](../README.md) y verificada en el [sitio publicado](https://rick1130042.github.io/Netflix_clasificadores/). |

## Antes de enviar

- [ ] Entregar en Canvas la presentación final y el enlace de la entrega.
- [ ] Probar desde una ventana privada las cuentas de gerencia y analista y compartirlas con el docente por un canal privado.
- [ ] Confirmar quién verifica el CSV y acordar el horizonte de la meta del piloto.
- [ ] Incorporar el mockup y los prompts originales si el equipo puede recuperarlos.
- [ ] Preparar la defensa individual con las métricas y sus límites.
