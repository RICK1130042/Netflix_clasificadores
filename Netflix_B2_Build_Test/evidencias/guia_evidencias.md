---
noteId: "f10c0160e15d11f18255d32709769dd3"
tags: []

---

# Lista de evidencias reales para la rúbrica

1. **Tipos en pantalla (15%)**: ejecutar `python scripts/integrar_app.py` y `streamlit run app.py`; abrir `🧪 Calidad del dato` y capturar el estado `DATOS VALIDADOS`, los 14 dtypes y la cuota mensual con números ordenables. Guardar captura en `evidencias/calidad_tipos.png`.
2. **Contrato (15%)**: revisar `contrato-datos.md`; el equipo debe ratificar significados de vacíos y unidades. En el contrato existe una fila por cada columna fuente y por cada columna de las tablas expuestas.
3. **Pipeline que falla (12%)**: hacer la prueba **en una rama**, siguiendo `GUIA_INSTALACION.md`, y copiar la **URL REAL** de la ejecución roja de GitHub Actions dentro de `README.md`. Nunca insertar una liga inventada ni enviar a `main` un CSV roto.
4. **Tabla con fuente (8%)**: capturar columnas `Cuota mensual · DATO`, `Fuente`, `Clientes · CALCULADO`, `Abandono (%) · CALCULADO`, con `column_order` y `hide_index=True` en código.
5. **Avance personal (12%)**: captura de la mejora incorporada, enlace al archivo/commit y explicación de cómo funciona. Completar `evidencias/avance_individual.md` con aportación real.
6. **Idea (8%)**: renombrar `ideas/tu-matricula.md` por matrícula y revisar que realmente sea la propuesta individual.
7. **Investigación (30%)**: `investigacion/claude.md` incluye los seis pasos, evidencia de prompts y tres cambios, al menos uno dentro y otro fuera de CI.
8. **Prompts:** `prompts/B2.md`, completar si el equipo usó otros asistentes o chats que no están disponibles en este proyecto.

**No confundir una prueba local negativa con la corrida de GitHub que exige el docente.** La primera se puede repetir con `python scripts/demo_contrato_roto.py`; la segunda necesita publicar el workflow y una rama de prueba.
