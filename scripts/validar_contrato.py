"""Salida no cero ante contrato inválido; no modifica el archivo de entrada."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validacion_datos import leer_y_validar

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    args = parser.parse_args()
    try:
        reporte = leer_y_validar(args.csv)
    except (ValueError, OSError, TypeError) as exc:
        print(f'CONTRATO INVÁLIDO: {type(exc).__name__}')
        return 1
    if not reporte.valido:
        # Sólo nombres/reglas: no imprimir filas ni valores de usuarios en CI.
        print('CONTRATO INVÁLIDO; revisar fuente localmente.')
        return 1
    print(f'CONTRATO VÁLIDO: {len(reporte.datos)} filas, {len(reporte.datos.columns)} columnas')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
