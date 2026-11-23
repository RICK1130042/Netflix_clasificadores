---
noteId: "2d73e0e0e76811f19ab549f7efe655a6"
tags: []

---

# Entrega de Enrique — Semanas 15 y 16

Archivos preparados para el repositorio `RICK1130042/Netflix_clasificadores`.

## Archivos

- `src/reentrenar.py`: semana 15, reentrenar y actualizar solo cuando el F1 suba.
- `docs/Prediction_y_Model_Updating.md`: semana 16, explicación, diagrama y preguntas de defensa.
- `data/nuevos_clientes_DEMO.csv`: ejemplo **sintético**, solo para probar el funcionamiento.

**Importante:** el repositorio usa `modelo_churn_netflix.joblib` con las claves `modelo` y `columnas`, y el dataset original `netflix_customer_churn.csv`. **No copies encima** de esos archivos al extraer este paquete.

## Paso 1 — Rama segura

Desde la terminal, en la raíz del proyecto:

```bash
git status
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c feature/enrique-semanas-15-16
git branch --show-current
```

Si `git status` muestra modificaciones pendientes, guárdalas antes de cambiar. Nunca ejecutes `git reset --hard` para este trabajo.

## Paso 2 — Copiar los archivos del ZIP

Extrae el ZIP dentro de `Netflix_clasificadores/`, respetando las carpetas `src/`, `docs/` y `data/`. Conserva tus archivos originales.

## Paso 3 — Preparar Python (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_streamlit.txt xgboost
```

Si ya tienes un entorno del proyecto, úsalo en lugar de crear otro. Para cargar un `.joblib` antiguo puede ser necesario utilizar las mismas versiones de scikit-learn y XGBoost con las que se generó.

## Paso 4 — Probar sin tocar el modelo

```bash
python src/reentrenar.py --nuevo data/nuevos_clientes_DEMO.csv
```

Muestra el F1 anterior, el F1 del candidato y la decisión. Como no lleva `--aplicar`, **no guarda nada**. Los valores dependen del modelo local; no hay números precalculados ni inventados.

## Paso 5 — Usar un CSV real cuando lo tengas

Debe tener al menos las columnas:

`customer_id, churned, watch_hours, last_login_days, avg_watch_time_per_day, number_of_profiles, subscription_type`

- `churned`: 0 o 1, sin nulos.
- `subscription_type`: Basic, Premium o Standard.
- `customer_id`: único en el CSV; no puede incluir clientes del conjunto reservado para evaluación.

```bash
python src/reentrenar.py --nuevo data/nuevos_clientes.csv
python src/reentrenar.py --nuevo data/nuevos_clientes.csv --aplicar
```

El segundo comando reemplaza `modelo_churn_netflix.joblib` **solo si aumenta F1**. En `main/master` se bloquea. No cargues archivos `.joblib` de procedencia desconocida.

## Paso 6 — Subir tu trabajo sin mezclar con main

```bash
git status
git add src/reentrenar.py docs/Prediction_y_Model_Updating.md README_Enrique.md data/nuevos_clientes_DEMO.csv
git commit -m "Entrega Enrique semanas 15 y 16"
git push -u origin feature/enrique-semanas-15-16
```

Si autorizaste y el script efectivamente actualizó el `.joblib`, revisa primero con el equipo si también desean subir ese archivo binario en tu rama. No uses `git add .` sin revisar los cambios. `main` no se modifica hasta hacer un merge.

## Para exponer en clase

1. Explica el diagrama de `docs/Prediction_y_Model_Updating.md`.
2. Corre el primer comando de prueba y muestra la comparación de F1.
3. Explica que `--aplicar` requiere mejora y prohíbe escritura desde `main`.
4. Aclara que el archivo DEMO es sintético y que una actualización real necesita nuevos clientes con etiqueta conocida.

**Nota técnica:** el repositorio contiene `app.py` y `app3.py`, con algunas diferencias en selección/exhibición de modelos. Este paquete no altera esas aplicaciones. El ganador por validación cruzada en `comparacion_validacion_cruzada.csv` es XGBoost; Random Forest es ligeramente superior en la comparación de prueba que figura en `comparacion_modelos_churn.csv`.
