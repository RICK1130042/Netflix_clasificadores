"""Comprueba numéricos de CSV adicionales de resultados publicados, sin mezclarlos.

No valida que las métricas sean estadísticamente correctas: valida estructura,
tipos, rangos y que cada fuente conserve su identificación.
"""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ESQUEMAS = {
    "comparacion_modelos_churn.csv": {"Accuracy": 1, "Precision": 1, "Recall": 1, "F1": 1, "ROC-AUC": 1},
    "comparacion_validacion_cruzada.csv": {"Accuracy": 1, "Precision": 1, "Recall": 1, "F1": 1, "ROC-AUC": 1, "Desviación F1": 1},
    "reports/generated/churn_por_segmento.csv": {"registros": None, "tasa_churn_pct": 100},
}


def validar_uno(path: Path, columnas: dict) -> list[str]:
    errores = []
    try:
        df = pd.read_csv(path, dtype="string", keep_default_na=False)
    except Exception as exc:
        return [f"{path.name}: imposible leer el archivo: {exc}"]
    if df.empty:
        return [f"{path.name}: archivo vacío"]
    for columna, maximo in columnas.items():
        if columna not in df:
            errores.append(f"{path.name}: falta '{columna}'")
            continue
        raw = df[columna].str.strip()
        valores = pd.to_numeric(raw, errors="coerce")
        if (valores.isna() | (raw == "")).any():
            errores.append(f"{path.name}/{columna}: hay datos no numéricos o ausentes")
            continue
        if not (valores >= 0).all() or (maximo is not None and not (valores <= maximo).all()):
            errores.append(f"{path.name}/{columna}: hay valores fuera de rango")
        if columna == "registros" and not (valores % 1 == 0).all():
            errores.append(f"{path.name}/{columna}: conteos fraccionarios")
    return errores


def main() -> int:
    errores = []
    for nombre, columnas in ESQUEMAS.items():
        ruta = ROOT / nombre
        if not ruta.is_file():
            errores.append(f"Falta el archivo publicado: {nombre}")
            continue
        errores.extend(validar_uno(ruta, columnas))
    for error in errores:
        print("ERROR:", error, file=sys.stderr)
    if errores:
        print("REPORTES INVÁLIDOS: se detuvo la verificación", file=sys.stderr)
        return 1
    print("REPORTES VÁLIDOS: tipos y rangos de exportaciones comprobados (sin mezclar experimentos).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
