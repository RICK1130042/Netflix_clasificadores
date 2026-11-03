"""Informe de calidad reproducible: nulos, duplicados, valores extremos y segmentos.

Uso: python scripts/reporte_calidad.py --input data/prepared/netflix_limpio.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def generar(ruta: Path, salida: Path) -> None:
    df = pd.read_csv(ruta)
    salida.mkdir(parents=True, exist_ok=True)
    perfil = pd.DataFrame({
        "variable": df.columns,
        "tipo_python": [str(df[c].dtype) for c in df.columns],
        "nulos": [int(df[c].isna().sum()) for c in df.columns],
        "porcentaje_nulos": [round(100 * df[c].isna().mean(), 2) for c in df.columns],
        "valores_unicos": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    perfil.to_csv(salida / "perfil_columnas_python.csv", index=False)
    extremos = []
    for c in df.select_dtypes(include="number").columns:
        if c == "churned":
            continue
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        n = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
        extremos.append({"variable": c, "regla": "1.5 x IQR", "casos_senialados": n})
    pd.DataFrame(extremos).to_csv(salida / "outliers_iqr.csv", index=False)

    segmentos = []
    if "churned" in df.columns:
        for campo in ["gender", "region", "subscription_type", "payment_method", "favorite_genre"]:
            if campo in df:
                agrupado = df.groupby(campo, dropna=False)["churned"].agg(["size", "mean"])
                for categoria, fila in agrupado.iterrows():
                    segmentos.append({"variable": campo, "categoria": str(categoria),
                                      "registros": int(fila["size"]),
                                      "tasa_churn_pct": round(float(fila["mean"])*100, 2)})
    pd.DataFrame(segmentos).to_csv(salida / "churn_por_segmento.csv", index=False)
    tasa = float(df["churned"].mean() * 100) if "churned" in df else float("nan")
    reporte = f"""# Calidad de datos — Netflix

- Filas: **{len(df)}**
- Columnas: **{len(df.columns)}**
- Duplicados exactos: **{df.duplicated().sum()}**
- Duplicados de customer_id: **{df['customer_id'].duplicated().sum() if 'customer_id' in df else 'No aplica'}**
- Tasa descriptiva de churn: **{tasa:.2f}%**

## Archivos
- `perfil_columnas_python.csv`: tipos y nulos por campo.
- `outliers_iqr.csv`: valores que ameritan revisión por regla IQR (no eliminación automática).
- `churn_por_segmento.csv`: distribución y tasas descriptivas por categoría.

## Precauciones de interpretación
- Un outlier estadístico no es necesariamente un error.
- Diferencias por grupo no establecen causalidad ni demuestran por sí solas sesgo del modelo.
- Para estudiar sesgo predictivo, medir también falsos negativos, recall y tamaños por grupo en test.
- Confirmar que las variables de actividad se observan antes del evento `churned`.
"""
    (salida / "reporte_calidad.md").write_text(reporte, encoding="utf-8")
    print(f"Reporte generado en {salida.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/prepared/netflix_limpio.csv"))
    p.add_argument("--output", type=Path, default=Path("reports/generated"))
    a = p.parse_args()
    generar(a.input, a.output)
