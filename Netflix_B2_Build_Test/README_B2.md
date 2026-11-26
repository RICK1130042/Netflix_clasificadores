> Estado auditado 26-nov-2026: esta guía contenía rutas y funcionalidades propuestas sin integrar. Consultar [auditoría actual](../docs/auditoria-entrega.md) y [corridas reales](../docs/evidencia-ci.md). No usar los comandos históricos de exportación/integración inexistentes.



## ✅ Entregable B2 — Build y Test: contrato de datos del dashboard

El contrato documentado en [`contrato-datos.md`](contrato-datos.md) cubre las columnas originales del CSV y las columnas publicadas por el dashboard, sus unidades, fuente, vacíos y distinción **DATO/CALCULADO/SIMULADO**. Las decisiones de metodología se explican en [`investigacion/claude.md`](investigacion/claude.md) y en [`CLAUDE.md`](CLAUDE.md).

### Verificaciones antes de presentar el dashboard

```bash
python scripts/validar_contrato.py --csv netflix_customer_churn.csv
python scripts/validar_exportaciones.py
python -m unittest discover -s tests -v
streamlit run app.py
```

El validador central reconoce **14 campos originales**, convierte columnas numéricas sin pérdida silenciosa y detiene el dibujo si encuentra errores. La nueva página **🧪 Calidad del dato** enseña el resultado y ofrece una tabla con cuota mensual en tipo decimal, ordenamiento numérico, `column_order` y `hide_index=True`.

Los controles de `.github/workflows/calidad-datos.yml` se ejecutan en cada `push` y `pull_request`. Para que GitHub **impida fusionar** un PR con checks rojos, configurar en la protección de la rama `main` el job requerido `contrato-de-datos`.

### Evidencia de un cambio roto (llenar después de ejecutarlo)

- **Rama/PR de prueba:** `prueba-contrato-roto` (crearla para esta demostración; no dañar `main`).
- **Error provocado:** texto `8.99 ERROR` en la columna `monthly_fee` de una fila del CSV original.
- **Corrida fallida de GitHub Actions:** **PENDIENTE — REEMPLAZAR POR URL REAL DEL RUN FALLIDO**.
- **Corrida corregida/aprobada:** **PENDIENTE — REEMPLAZAR POR URL REAL DEL RUN APROBADO**.
- **Captura de la aplicación con estado del contrato:** **PENDIENTE — ver `evidencias/guia_evidencias.md`**.

Acceso al historial del workflow después de publicarlo: https://github.com/RICK1130042/Netflix_clasificadores/actions/workflows/calidad-datos.yml

### Procedencia de métricas y limitaciones

El CSV original del repositorio tiene 5,000 filas y 14 columnas. El PDF de la presentación muestra un experimento de Random Forest y capturas de SAS con denominadores distintos; **no combinar cifras de SAS y Python sin verificar filtros y definición**. En la app actual, Gradient Boosting fue elegido mediante validación cruzada en su experimento; la presentación original destacó Random Forest. Son resultados reportados de análisis diferentes. La app no demuestra impacto causal de intervenciones ni pronostica abandono temporal futuro sin validar fechas.

### Evidencia individual

La propuesta individual se encuentra en [`ideas/tu-matricula.md`](ideas/tu-matricula.md) (**renombrar este archivo por la matrícula real**). El avance de producto, captura y enlace de commit se documenta en [`evidencias/avance_individual.md`](evidencias/avance_individual.md).
