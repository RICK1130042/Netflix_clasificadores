# Seguimiento de observaciones del 21 de octubre

> Registro histórico de la primera revisión. La [entrega final](entrega-final.md) usa un CSV derivado validado. El hallazgo de la fuente original y las corridas históricas se conservan como evidencia.

## O-1 — fuente: retroalimentación docente, criterio C
Qué observaron: demostrar que los roles funcionan.
Qué cambió: `tests/test_roles.py` ejecuta auth.py en Streamlit AppTest, inicia sesión con ambos roles y recorre sus páginas, rechaza credenciales/rol inválidos, fuerza acceso prohibido y prueba salida/cambio de rol. `tests/test_tabla.py` verifica el orden del control de acceso en app.py.
Dónde quedó: los archivos anteriores y las corridas enlazadas en `docs/evidencia-ci.md`. Prueba aislada de autenticación; pendiente captura de la app completa en Cloud y contraste con la Investigación 3 y propuesta de octubre. No se declara cierre completo de O-1.

## O-2 — fuente: retroalimentación docente, criterio de publicación del CSV
Qué observaron: declarar quién verifica el CSV antes de publicar y cómo.
Qué cambió: workflow raíz valida fuente y preparado, CLI falla sin modificar archivos, y pruebas negativas cubren datos sintéticos. Se detectaron 10 valores diarios superiores a 24; publicación pendiente de resolverlos contra fuente.
Dónde quedó: `.github/workflows/calidad-datos.yml`, `scripts/validar_contrato.py`, `validacion_datos.py`, `tests/test_validacion_datos.py`, `tests/test_contrato_cli.py` y `docs/evidencia-ci.md`. El equipo debe nombrar responsable y revisor y registrar su aprobación; no se atribuyen nombres sin confirmación.

Procedimiento propuesto: responsable conserva fuente y hash, ejecuta ambas validaciones y revisa errores; revisor coteja procedencia, ausencias, unidades y cambios; adjuntan corrida al PR antes de autorizar publicación. Designación de personas y aprobación de excepciones siguen pendientes.
