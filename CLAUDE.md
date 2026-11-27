# Convenciones del repositorio

- Mantener la app raíz y las pruebas de roles, tabla y contraste. No versionar secretos.
- Validar con `python scripts/preparar_entrega.py --check`, `python scripts/validar_contrato.py --csv datos/netflix_validado.csv` y `python -m unittest discover -s tests -v`.
- Distinguir datos fuente, cálculos y simulaciones. Separar métricas históricas exportadas y análisis de la cohorte activa.
- Registrar prompts por entregable con texto y procedencia cuando existan. Identificar cambios y decisiones humanas verificables.
- Revisar datos y accesibilidad antes de desplegar. Señalar registros fuera de contrato sin corregirlos por intuición.
