# B2 — Contrato de datos y evidencia ejecutable

El contrato vive en [contrato-datos.md](contrato-datos.md); investigación existente: [investigacion/claude.md](investigacion/claude.md). Revisión actual: [auditoría](../docs/auditoria-entrega.md) y [corridas reales](../docs/evidencia-ci.md).

Desde la raíz del repositorio:

```sh
python scripts/validar_contrato.py --csv netflix_customer_churn.csv
python scripts/validar_contrato.py --csv data/prepared/netflix_limpio.csv
python -m unittest discover -s tests -v
```

El workflow ahora vive en `.github/workflows/calidad-datos.yml` raíz. Usa el validador raíz y las pruebas de tests/. La copia del validador en esta subcarpeta queda como evidencia histórica B2, no es la que ejecuta CI.

La suite sintética y de roles pasa; fuente/preparado incumplen el máximo de 24 horas en 10 valores de avg_watch_time_per_day. No hay aprobación integral de datos. La app raíz conserva su validador reducido y no tiene una página Calidad del dato. No existen scripts/validar_exportaciones.py ni scripts/integrar_app.py en esta versión.

Las demostraciones roja/reversión usan únicamente tests/fixtures/contrato_sintetico.csv y palette.py en la rama prueba/ci-negativo-2026-11-26. No alteran el CSV publicado ni main. Las corridas, commits y límites de evidencia están en docs/evidencia-ci.md.

Las propuestas individuales existentes están en ideas/ y los avances en evidencias/. Para Ricardo: [ideas/A01666859.md](ideas/A01666859.md) y [evidencias/avance_A01666859.md](evidencias/avance_A01666859.md). No se cambian la propuesta ni las decisiones atribuidas a integrantes.

La fuente de métricas de Random Forest del README y la selección posterior del dashboard representan experimentos distintos; no se combinan cifras de SAS/Python ni se afirman efectos causales de campañas.
