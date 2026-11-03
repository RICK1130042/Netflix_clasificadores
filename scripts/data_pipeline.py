"""Preparación reproducible del dataset público Netflix.

Uso desde la raíz del proyecto:
  python scripts/data_pipeline.py --input netflix_customer_churn.csv

IMPORTANTE: los campos de actividad solo deben usarse como predictores si
su periodo de medición es anterior a churned (evita fuga de información).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

COLUMNAS_ORIGINALES = [
    "customer_id", "age", "gender", "subscription_type", "watch_hours",
    "last_login_days", "region", "device", "monthly_fee", "churned",
    "payment_method", "number_of_profiles", "avg_watch_time_per_day",
    "favorite_genre",
]
NUMERICAS = ["age", "watch_hours", "last_login_days", "monthly_fee",
             "number_of_profiles", "avg_watch_time_per_day", "churned"]
CATEGORICAS = ["gender", "subscription_type", "region", "device",
              "payment_method", "favorite_genre"]
DESCRIPCIONES = {
    "customer_id": "Identificador de cliente; excluir del entrenamiento",
    "age": "Edad del cliente",
    "gender": "Género declarado",
    "subscription_type": "Tipo de suscripción",
    "watch_hours": "Horas de visualización; verificar ventana temporal",
    "last_login_days": "Días desde el último inicio de sesión; verificar ventana temporal",
    "region": "Región",
    "device": "Dispositivo",
    "monthly_fee": "Cuota mensual",
    "churned": "Variable objetivo: 0 permanece; 1 cancela",
    "payment_method": "Método de pago",
    "number_of_profiles": "Número de perfiles",
    "avg_watch_time_per_day": "Tiempo medio diario; candidato a análisis de fuga/sobreajuste",
    "favorite_genre": "Género favorito de contenido",
    "segmento_inactividad": "Derivada de last_login_days: 0-7, 8-30 o 31+ días",
}


def ejecutar(archivo_entrada: Path, salida: Path) -> dict:
    if not archivo_entrada.is_file():
        raise FileNotFoundError(f"No se encontró el dataset: {archivo_entrada}")
    df = pd.read_csv(archivo_entrada)
    faltantes = sorted(set(COLUMNAS_ORIGINALES) - set(df.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas del esquema original: {faltantes}")

    filas_iniciales = len(df)
    duplicados_exactos = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    for c in CATEGORICAS:
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA)
    df["customer_id"] = df["customer_id"].astype("string").str.strip()
    for c in NUMERICAS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    claves_duplicadas = int(df["customer_id"].duplicated().sum())
    if claves_duplicadas:
        raise ValueError(
            f"Existen {claves_duplicadas} customer_id duplicados después de "
            "quitar filas exactas; revisa los registros antes de continuar."
        )
    valores_objetivo = set(df["churned"].dropna().unique().tolist())
    if not valores_objetivo.issubset({0, 1}) or df["churned"].isna().any():
        raise ValueError("'churned' debe contener solamente 0 y 1, sin nulos")
    df["churned"] = df["churned"].astype(int)

    # Feature engineering descriptivo: corte fijo, no aprendido del target.
    df["segmento_inactividad"] = pd.cut(
        df["last_login_days"],
        bins=[-float("inf"), 7, 30, float("inf")],
        labels=["0-7 días", "8-30 días", "31+ días"],
    ).astype("string")
    salida.mkdir(parents=True, exist_ok=True)
    ruta_csv = salida / "netflix_limpio.csv"
    ruta_parquet = salida / "netflix_limpio.parquet"
    ruta_dicc = salida / "diccionario_variables.csv"
    df.to_csv(ruta_csv, index=False)
    df.to_parquet(ruta_parquet, index=False)

    dicc = pd.DataFrame([
        {"variable": c, "tipo_python": str(df[c].dtype),
         "rol": "objetivo" if c == "churned" else (
             "identificador" if c == "customer_id" else (
                 "derivada descriptiva" if c == "segmento_inactividad" else "predictora candidata")),
         "descripcion": DESCRIPCIONES.get(c, "Variable adicional")}
        for c in df.columns
    ])
    dicc.to_csv(ruta_dicc, index=False)
    resumen = {
        "origen": str(archivo_entrada),
        "filas_iniciales": filas_iniciales,
        "filas_finales": int(len(df)),
        "columnas_finales": int(df.shape[1]),
        "duplicados_exactos_removidos": duplicados_exactos,
        "ids_duplicados": claves_duplicadas,
        "nulos_por_columna": {str(k): int(v) for k, v in df.isna().sum().items()},
        "churn_1": int((df["churned"] == 1).sum()),
        "nota": "No se imputan valores aquí: la imputación del modelo debe ajustarse solo con train.",
    }
    (salida / "resumen_pipeline.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumen


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("netflix_customer_churn.csv"))
    p.add_argument("--output", type=Path, default=Path("data/prepared"))
    args = p.parse_args()
    print(json.dumps(ejecutar(args.input, args.output), indent=2, ensure_ascii=False))
