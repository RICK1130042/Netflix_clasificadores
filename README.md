[Dashboard en Streamlit Cloud](https://rick1130042-netflix-clasificadores-app-rzfx7h.streamlit.app/)

# Netflix · Customer Churn Analytics · Equipo 1

El tablero permite identificar segmentos con mayor abandono y preparar una prueba de retención. La población activa del tablero contiene 4,990 registros académicos y 2,515 abandonos observados (50.4%). La [meta del piloto](negocio.md) está documentada como propuesta.

**Evaluación.** Abra el dashboard en ventana privada. La aplicación tiene cuentas de demostración para gerencia y analista. El equipo comparte esas credenciales de forma privada. La presentación final se entrega en Canvas.

**Documentación viva.** [Sitio de la entrega](https://rick1130042.github.io/Netflix_clasificadores/) · [Pipeline de publicación](https://github.com/RICK1130042/Netflix_clasificadores/actions/workflows/documentacion.yml).

| Artefacto | Dónde encontrarlo |
|---|---|
| App, dependencias y datos | [app.py](app.py), [requirements.txt](requirements.txt), [datos/](datos/) |
| Decisión, línea base y meta | [negocio.md](negocio.md) |
| Contratos | [Prompt](contrato-prompt.md) · [Datos](contrato-datos.md) |
| Gráficas, jerarquía y accesibilidad | [Fichas](graficas.md) · [Evidencia B3](docs/entregable-3.md) |
| Acceso y datos sensibles | [Matriz](acceso.md) · [Pruebas de roles](tests/test_roles.py) |
| Prompts | [B1](prompts/B1.md) · [B2](prompts/B2.md) · [B3](prompts/B3.md) · [B4](prompts/B4.md) · [Análisis por etapa](prompts/analisis-sdlc.md) |
| Investigaciones | [SDLC](investigacion/00-sdlc-genai.md) · [Jerarquía](investigacion/01-jerarquia.md) · [Contraste](investigacion/02-contraste.md) · [Acceso](investigacion/03-acceso.md) · [Métricas](investigacion/04-metricas.md) |
| Paleta y convenciones | [estilos.py](estilos.py) · [CLAUDE.md](CLAUDE.md) |
| Evolución del diseño | [Design thinking y capturas](docs/design-thinking.md) |
| Regresión y retroalimentación | [Corridas](docs/evidencia-ci.md) · [Observaciones de octubre](docs/observaciones-21-oct.md) · [Estado final](docs/entrega-final.md) |

## Verificación local

```sh
python -m pip install -r requirements.txt
python scripts/preparar_entrega.py --check
python scripts/validar_contrato.py --csv datos/netflix_validado.csv
python -m unittest discover -s tests -v
streamlit run app.py
```

El proyecto usa secretos locales o de Streamlit Cloud para las cuentas. Las pruebas usan cuentas efímeras propias. Consulte el [contrato](contrato-datos.md) para reproducir la preparación y distinguir las métricas históricas de la evaluación sobre el CSV activo.
