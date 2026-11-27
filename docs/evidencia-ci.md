# Evidencia CI — 26 de noviembre de 2026

> Registro histórico de la primera revisión. La [entrega final](entrega-final.md) usa un CSV derivado validado. El hallazgo de la fuente original y las corridas históricas se conservan como evidencia.

Repositorio: RICK1130042/Netflix_clasificadores. Base main: `23585afd949a2adba573bd8b74bfb82e3630e497`. Rama de entrega: `revision/entrega-evidencias-2026-11-26`. Rama de demostración: `prueba/ci-negativo-2026-11-26` (no fusionar).

## Pruebas de roles, tabla y regresión

Implementación `671b431535139e9613a34a75bd368ff114f184fe`: [job aprobado, 26 pruebas](https://github.com/RICK1130042/Netflix_clasificadores/actions/runs/45913842180/job/140228233514). Logs comprobados: `Ran 26 tests`, `OK`. También 26 pruebas locales aprobadas con Python 3.13; CI usa Python 3.12. AppTest ejecuta auth.py con secretos efímeros, sin ejecutar modelos ni datos del dashboard completo. No equivale a capturas de roles en Cloud.

## Contrato de datos: fallo deliberado y reversión

| Estado | Commit | Evidencia comprobada |
|---|---|---|
| Rojo deliberado: monthly_fee sintético = 8.99 ERROR | `9f8de7298cd4b2aa6b4cdfb374dc2ee526869a03` | [Job fallido](https://github.com/RICK1130042/Netflix_clasificadores/actions/runs/45913862866/job/140228301485), paso «Fixture sintético para demostración aislada» falla |
| Revertido: fixture original restaurado | `adc6cf1013290fb7700b93dea5fa2dc17063a4d2` | [Job aprobado](https://github.com/RICK1130042/Netflix_clasificadores/actions/runs/45913928726/job/140228517554), fixture y 26 pruebas pasan |
| Estado final de rama de prueba | `ae6a88a7c8314bd298b463a4f44fca66f2c65d9f` | [Job aprobado](https://github.com/RICK1130042/Netflix_clasificadores/actions/runs/45913992999/job/140228735880) |

La corrupción solo afectó tests/fixtures/contrato_sintetico.csv (dos filas artificiales). No se tocaron los CSV publicados. Los workflows completos de datos siguen rojos por el job fuente-publicada; los enlaces verdes anteriores son de jobs sintéticos/regresión, no aprobación integral de datos.

## Contraste: fallo deliberado y reversión

| Estado | Commit | Evidencia comprobada |
|---|---|---|
| Implementación de entrega en verde | `671b431535139e9613a34a75bd368ff114f184fe` | [Corrida verde](https://github.com/RICK1130042/Netflix_clasificadores/actions/runs/45913842046) |
| Rojo deliberado: texto del mismo color que fondo | `a29004eb5db710b614237083aee885506dfe3e5e` | [Corrida roja](https://github.com/RICK1130042/Netflix_clasificadores/actions/runs/45913949554), job 140228587729; logs: texto/fondo 1.000:1, mínimo 4.5 |
| Paleta original restaurada | `ae6a88a7c8314bd298b463a4f44fca66f2c65d9f` | [Corrida verde](https://github.com/RICK1130042/Netflix_clasificadores/actions/runs/45913992969) |

No se modifican umbrales ni se eliminan pruebas para lograr verde. El commit final de la rama de demostración restaura exactamente el árbol de la implementación de entrega; las dos alteraciones quedan solo en historial de esa rama.

## Validación de fuente: problema real todavía abierto

[Job fuente-publicada rojo de la rama de entrega](https://github.com/RICK1130042/Netflix_clasificadores/actions/runs/45913842180/job/140228233814). Fallan ambos archivos: netflix_customer_churn.csv y data/prepared/netflix_limpio.csv, con 10 valores avg_watch_time_per_day >24. No se relaja el contrato ni se recortan valores para ocultar el resultado. La línea base recalculada es 2,515/5,000 =50.3%; no es el 23% del ejemplo del profesor.

Reproducir: `python scripts/validar_contrato.py --csv netflix_customer_churn.csv` y lo mismo con `data/prepared/netflix_limpio.csv`. Salida esperada actual: 1. El equipo debe contrastar la definición/unidad con la fuente, corregir con trazabilidad o aprobar justificadamente una modificación del contrato, y luego obtener una corrida integral verde.

## Preservación y seguridad

main se volvió a consultar y conserva `23585afd949a2adba573bd8b74bfb82e3630e497`. No se cambia rama/configuración de Streamlit, tema, auth, modelos ni CSV de la rama de entrega. URL declarada conservada: https://rick1130042-netflix-clasificadores-app-rzfx7h.streamlit.app/. No se afirma comprobación visual en ventana privada ni monitoreo del despliegue. Antes/después de la prueba es evidencia por ramas/archivos, pendiente captura Cloud para el profesor.

SHA-256 de entradas preservadas:

| Archivo | SHA-256 |
|---|---|
| netflix_customer_churn.csv | 16e00950956f12921625001315dff2c450c9ca14af885cb19b4035b0b6fb311e |
| data/prepared/netflix_limpio.csv | 6884eb7c7153693f2bb07e332d8694c40103e29903f1e8ed4562c399a235df07 |
| modelo_churn_netflix.joblib | 2ec2c9b9e61840b4b90ba24e59cbeec68fe0c26d24eed90e57b96626a51d61a8 |

No se detectaron archivos secrets.toml/.env versionados ni patrones de claves privadas, GitHub, AWS u OpenAI en texto del árbol revisado. No se garantiza ausencia de secretos en todo el historial. No se publicaron credenciales aportadas en conversaciones.
