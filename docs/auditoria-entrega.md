# Auditoría de entrega — 26 de noviembre de 2026

> Registro histórico de la primera revisión. La [entrega final](entrega-final.md) usa un CSV derivado validado. El hallazgo de la fuente original y las corridas históricas se conservan como evidencia.

Base revisada: main `23585afd949a2adba573bd8b74bfb82e3630e497`. Retroalimentación completa recuperada del último mensaje de la conversación «Entregar repositorio GitHub». Cambios preparados en rama independiente. No se modifica main, configuración de Cloud, modelo, datasets, pipeline de preparación ni credenciales del host.

| Requisito | Hallazgo y cambio verificable | Pendiente del equipo |
|---|---|---|
| Despliegue | README ya tiene URL en primera línea; se conserva | Confirmar ventana privada. El requisito «sin sesión» entra en tensión con login por roles: acordar criterio con docente, sin quitar login unilateralmente |
| Negocio | Se incorpora negocio.md como propuesta y línea base reproducible 50.3% | Aprobar persona, periodicidad y meta; 23% del ejemplo docente no coincide con CSV |
| B1 | No existe historial B1 en main | Recuperar prompts literales, fechas, modelo, cambios y decisiones propias; no reconstruir de memoria como cita literal |
| B2/prompts | Hay extractos y tareas resumidas en Netflix_B2_Build_Test/prompts/B2.md | No son un registro literal completo por interacción; confirmar autoría |
| B3 | Se conserva el único prompt documentado y se explicitan campos desconocidos | Validar ajustes y decisiones del equipo; no declarar validado por cambiar el título |
| Investigación 1 | No se encontró documento en el árbol de main | Aportar autor, título, año, URL, hallazgo y decisión visual derivada |
| Investigación 3 | No se encontró documento en el árbol de main | Aportar investigación y matriz de permisos de octubre; cotejar implementación |
| Observaciones 21 oct | Nuevo seguimiento y pruebas reproducibles | Nombrar verificador/revisor CSV y evidencia Cloud de roles |
| Contrato | Documento B2 conserva seis secciones; CLI raíz reutiliza reglas | 10 valores avg_watch_time_per_day >24 impiden aprobar fuente y preparado; confirmar unidad/definición o corregir desde fuente con trazabilidad |
| Workflow datos | Movido a raíz; eliminadas llamadas a scripts inexistentes y dependencia de requirements_streamlit.txt inexistente | Configurar checks requeridos al decidir integrar, sin cambiar protección ahora |
| Contraste | Se mantienen fórmula, tema y umbrales; demostración negativa aislada | Revisión visual de accesibilidad completa sigue pendiente |
| Tabla | app.py: tabla por segmento declara hide_index y column_order y rótulo CALCULADO/fuente | Captura visual al revisar PR |

## Límites de la documentación previa

El README B2 y contrato describían una página «Calidad del dato» y scripts de exportación/integración que no existen en main. La app usa su propio validador reducido, separado del contrato B2. No se integra aquí la barrera de 14 campos en app.py porque detendría la app con los datos actuales; se preserva la base funcional. CI sí denuncia el incumplimiento; CI rojo no bloquea por sí solo un merge o despliegue sin reglas requeridas.

Los duplicados netflix_entregable3/ son un paquete histórico, no la app raíz; se señalan sus docs para evitar entregar una ficha vieja. No se reemplazan las investigaciones ausentes con bibliografía genérica. La investigación B2 existente recibe valoración positiva del profesor, pero su bibliografía y el PDF citado no han sido verificados independientemente aquí; ese PDF no está versionado.

## Seguridad

Revisión de árbol y texto versionado, incluidos patrones comunes de tokens/claves; no equivale a una auditoría exhaustiva de todo el historial o infraestructura. .gitignore excluye secretos de Streamlit y archivos .env; no se agregan credenciales del chat. Las cuentas en tests son efímeras y no son accesos Cloud. Auth usa secretos del host, rechaza rol desconocido y verifica permisos en servidor. No protege los CSV de un repositorio público y no tiene limitación persistente de intentos/MFA; confirmar carácter sintético y autorización de publicación de datos.

No se cargan modelos joblib durante las pruebas: pueden ejecutar código al deserializarse y solo deben cargarse si su origen es confiable. No se modifica despliegue ni se afirma revisión visual en Cloud. No se publica el historial de conversación que contenía contraseñas de demostración.

## Revisión y PR sin fusionar

1. Revisar diff contra main; confirmar que app.py cambia únicamente orden/rótulo de una tabla, y que CSV/modelo/tema/auth no cambian en la rama de entrega.
2. Ejecutar `python -m unittest discover -s tests -v` y ambos comandos CLI documentados. La suite debe pasar; la fuente seguirá roja hasta que el equipo resuelva los 10 registros.
3. Consultar docs/evidencia-ci.md y distinguir fallos deliberados de incumplimientos reales. Nunca fusionar la rama de demostración ni sus commits rotos.
4. Completar pendientes académicos con evidencia original y aprobación de personas responsables.
5. Abrir PR de la rama revision/entrega-evidencias-2026-11-26 hacia main, revisar checks y capturas. Mantenerlo sin fusionar hasta aprobación del equipo. No cambiar rama de Cloud.
