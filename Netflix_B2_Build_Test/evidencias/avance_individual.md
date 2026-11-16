---
noteId: "f10c0165e15d11f18255d32709769dd3"
tags: []

---

# Evidencia individual de avance — Entregable 2

**Persona:** Ricardo Villarreal (confirmar matrícula y compromiso que aparece en la hoja de concentración).  
**Fechas indicadas por el docente:** parte individual 13-nov-2026; partes de equipo 15-nov-2026.  
**Estado:** plantilla de evidencia preparada, **pendiente de captura y commit reales**. No afirmar que se entregó a tiempo sin registro.

## Opción A. Si el compromiso era revisar la calidad de datos / dashboard

1. **Qué queda en el producto:** `validacion_datos.py` contiene el contrato de tipos, `scripts/integrar_app.py` incorpora el panel en `app.py` y `tests/test_validacion_datos.py` prueba fallos reproducibles.
2. **Cómo funciona:** al cargar el CSV, la validación transforma las columnas numéricas a tipos Pandas reales y comprueba valores faltantes, rangos y categorías obligatorias; si hay errores, `st.error` enseña la causa y `st.stop()` evita dibujar gráficas. Los tests negativos requieren que datos rotos sean rechazados.
3. **Mi contribución real:** completar después de ejecutar y revisar los cambios: `[describir qué modificaste, revisaste o probaste tú; no atribuirte el trabajo automatizado de otro]`.
4. **Evidencia funcionando:** captura de `🧪 Calidad del dato`: `[ruta de imagen]`; commit o archivo GitHub: `[URL REAL]`.

## Opción B. Si el compromiso era de otra materia o no había un compromiso registrado

El docente pide cuatro puntos sobre el trabajo web del equipo. Reutiliza la explicación técnica de los puntos 1 y 2. **Después de integrar:** escribe tu contribución verificable y prepara una respuesta a una pregunta, por ejemplo: *"¿Por qué `monthly_fee` no puede mantenerse como texto?"* **Respuesta:** porque se ordenaría alfabéticamente en vez de numéricamente y podría generar operaciones incorrectas. Se convierte a `float` solo cuando todos los valores son válidos; un texto no convertible genera un error visible y bloquea la pantalla.

> No inventar la liga del commit, el compromiso personal ni una captura. Añadirlos al momento de realizar la prueba real.
