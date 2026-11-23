"""Semana 15 - Enrique: actualizar un modelo solo si mejora el F1.

Desde la raíz del repositorio:
    python src/reentrenar.py --nuevo data/nuevos_clientes_DEMO.csv
    python src/reentrenar.py --nuevo data/nuevos_clientes.csv --aplicar

Se adapta al archivo modelo_churn_netflix.joblib del proyecto Netflix.
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[1]
MODELO = RAIZ.parent / "modelo_churn_netflix.joblib"
ORIGINAL = RAIZ.parent / "netflix_customer_churn.csv"
NUMERICAS = [
    "watch_hours", "last_login_days", "avg_watch_time_per_day", "number_of_profiles"
]
PLANES = ["Basic", "Premium", "Standard"]
COLUMNAS = NUMERICAS + [f"subscription_type_{plan}" for plan in PLANES]
REQUERIDAS = ["customer_id", "churned", "subscription_type", *NUMERICAS]


def leer_y_validar(ruta):
    """Exige columnas, identificadores únicos, datos completos y target binario."""
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el CSV: {ruta}")
    df = pd.read_csv(ruta)
    faltantes = sorted(set(REQUERIDAS) - set(df.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas en {ruta.name}: {faltantes}")
    if df.empty or df[REQUERIDAS].isna().any().any():
        raise ValueError(f"{ruta.name}: no se permiten nulos ni CSV vacío")
    if df["customer_id"].duplicated().any():
        raise ValueError(f"{ruta.name}: hay customer_id repetidos")
    if not df["subscription_type"].isin(PLANES).all():
        raise ValueError(f"{ruta.name}: plan desconocido (Basic, Premium o Standard)")
    df = df.copy()
    for columna in NUMERICAS + ["churned"]:
        df[columna] = pd.to_numeric(df[columna], errors="raise")
    if not df["churned"].isin([0, 1]).all():
        raise ValueError("churned solo puede ser 0 o 1")
    if not np.isfinite(df[NUMERICAS].to_numpy(dtype=float)).all():
        raise ValueError("Hay valores numéricos inválidos")
    return df


def preparar_x(df, columnas_modelo):
    """Aplica el mismo One-Hot Encoding y orden que usa el dashboard."""
    x = pd.get_dummies(
        df[NUMERICAS + ["subscription_type"]],
        columns=["subscription_type"], dtype=int
    )
    return x.reindex(columns=columnas_modelo, fill_value=0)


def comprobar_rama():
    """Nunca permitir sobrescritura si seguimos en main/master."""
    proceso = subprocess.run(
        ["git", "branch", "--show-current"], cwd=RAIZ,
        capture_output=True, text=True, check=True
    )
    rama = proceso.stdout.strip()
    if not rama or rama in {"main", "master"}:
        raise RuntimeError("No se puede aplicar en main/master. Crea tu rama primero.")
    print(f"Rama Git verificada: {rama}")


def ejecutar(nuevo_csv, aplicar=False):
    if not MODELO.is_file():
        raise FileNotFoundError(f"No se encontró el modelo original: {MODELO}")

    original = leer_y_validar(ORIGINAL)
    nuevos = leer_y_validar(nuevo_csv)
    # Conservamos la partición del proyecto: 80 % entrenamiento, 20 % evaluación.
    base_train, evaluacion = train_test_split(
        original, test_size=0.20, stratify=original["churned"], random_state=42
    )

    # Impedir que registros de evaluación entren en el nuevo modelo.
    filtrados = set(nuevos["customer_id"]) & set(evaluacion["customer_id"])
    if filtrados:
        raise ValueError(
            f"El CSV nuevo contiene {len(filtrados)} clientes del conjunto de "
            "evaluación. No se puede reentrenar por fuga de datos."
        )

    # Si un cliente de entrenamiento reaparece, conservamos su versión nueva.
    entrenamiento = pd.concat([base_train, nuevos], ignore_index=True)
    entrenamiento = entrenamiento.drop_duplicates("customer_id", keep="last")

    # Cargar solamente el modelo generado por el equipo; joblib ejecuta pickle.
    paquete = joblib.load(MODELO)
    if not isinstance(paquete, dict) or not {"modelo", "columnas"} <= set(paquete):
        raise ValueError("El .joblib no contiene las claves 'modelo' y 'columnas'")
    columnas_modelo = list(paquete["columnas"])
    if set(columnas_modelo) != set(COLUMNAS):
        raise ValueError("Las columnas del modelo no coinciden con las del proyecto")

    viejo = paquete["modelo"]
    candidato = clone(viejo)  # Conserva hiperparámetros, borra entrenamiento anterior.
    x_train = preparar_x(entrenamiento, columnas_modelo)
    x_eval = preparar_x(evaluacion, columnas_modelo)
    candidato.fit(x_train, entrenamiento["churned"].astype(int))

    # Comparación justa: ambos modelos sobre exactamente los mismos clientes.
    y_eval = evaluacion["churned"].astype(int)
    f1_viejo = f1_score(y_eval, viejo.predict(x_eval), pos_label=1)
    f1_nuevo = f1_score(y_eval, candidato.predict(x_eval), pos_label=1)
    mejora = f1_nuevo > f1_viejo + 1e-12

    print(f"Modelo: {paquete.get('nombre_modelo', type(viejo).__name__)}")
    print(f"Registros del CSV nuevo: {len(nuevos)}")
    print(f"Registros para reentrenar: {len(entrenamiento)}")
    print(f"F1 anterior: {f1_viejo:.4f}")
    print(f"F1 candidato: {f1_nuevo:.4f}")
    print(f"Decisión: {'MEJORA' if mejora else 'NO MEJORA'}")

    if not mejora:
        print("Se conserva el modelo anterior. No se modificó ningún archivo.")
        return
    if not aplicar:
        print("Modo prueba: mejora detectada, pero no se guardó el modelo.")
        print("Para autorizar el cambio, ejecuta de nuevo con --aplicar.")
        return

    comprobar_rama()
    actualizado = dict(paquete)
    actualizado["modelo"] = candidato
    actualizado["actualizacion_enrique"] = {
        "metrica": "F1 para churned=1",
        "f1_anterior": float(f1_viejo),
        "f1_nuevo": float(f1_nuevo),
        "nuevas_filas": int(len(nuevos)),
        "evaluacion": "20 % del CSV original, random_state=42, estratificado",
    }

    # Reemplazo atómico: no se toca el archivo anterior si falla el guardado.
    with tempfile.NamedTemporaryFile(dir=MODELO.parent, suffix=".joblib", delete=False) as tmp:
        temporal = Path(tmp.name)
    try:
        joblib.dump(actualizado, temporal)
        os.replace(temporal, MODELO)
    finally:
        temporal.unlink(missing_ok=True)
    print(f"Modelo actualizado correctamente: {MODELO.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reentrenamiento seguro para Netflix")
    parser.add_argument("--nuevo", type=Path, required=True, help="CSV con clientes etiquetados")
    parser.add_argument("--aplicar", action="store_true", help="Guardar solo si F1 mejora")
    args = parser.parse_args()
    ruta_nueva = args.nuevo if args.nuevo.is_absolute() else RAIZ / args.nuevo
    ejecutar(ruta_nueva, aplicar=args.aplicar)
