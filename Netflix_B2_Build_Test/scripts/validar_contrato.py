"""Verificación bloqueante del CSV del tablero. Sale 1 si el contrato falla.

Uso: python scripts/validar_contrato.py --csv netflix_customer_churn.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Permite ejecutar el script directamente desde la raíz del repositorio.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validacion_datos import leer_y_validar  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=ROOT / "netflix_customer_churn.csv")
    args = p.parse_args()
    if not args.csv.is_file():
        print(f"ERROR: falta el CSV requerido: {args.csv}", file=sys.stderr)
        return 1
    try:
        reporte = leer_y_validar(args.csv)
    except Exception as exc:
        print(f"ERROR: no se pudo leer/validar: {exc}", file=sys.stderr)
        return 1
    print(reporte.detalle.to_string(index=False))
    for mensaje in reporte.advertencias:
        print(f"ADVERTENCIA: {mensaje}")
    for mensaje in reporte.errores:
        print(f"ERROR: {mensaje}", file=sys.stderr)
    if not reporte.valido:
        print("CONTRATO ROTO: se detiene el pipeline y no debe dibujarse el tablero.", file=sys.stderr)
        return 1
    print(f"CONTRATO VÁLIDO: {len(reporte.datos)} registros; campos numéricos correctamente tipados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
