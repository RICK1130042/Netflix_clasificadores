"""Genera el CSV de entrega sin imputar valores fuera del rango documentado."""
import argparse
import csv
import hashlib
import io
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "netflix_customer_churn.csv"
TARGET = ROOT / "datos" / "netflix_validado.csv"
REPORT = ROOT / "datos" / "preflight.json"
FIELD = "avg_watch_time_per_day"


def preparar():
    source = SOURCE.read_bytes()
    reader = csv.DictReader(io.StringIO(source.decode("utf-8-sig"), newline=""))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    total = accepted = excluded = cancellations = 0
    for row in reader:
        total += 1
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Fila {total}: número incorrecto de campos")
        try:
            daily_hours = Decimal(row[FIELD])
        except InvalidOperation:
            daily_hours = None
        if daily_hours is not None and daily_hours.is_finite() and daily_hours > 24:
            excluded += 1
            continue
        writer.writerow(row)
        accepted += 1
        cancellations += int(row["churned"] == "1")
    report = {
        "sha256_fuente": hashlib.sha256(source).hexdigest(),
        "filas_fuente": total,
        "filas_aceptadas": accepted,
        "filas_excluidas_horas_diarias_mayores_24": excluded,
        "abandonos_aceptados": cancellations,
    }
    return output.getvalue().encode(), (json.dumps(report, indent=2) + "\n").encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    dataset, report = preparar()
    for path, content in [(TARGET, dataset), (REPORT, report)]:
        if args.check:
            if path.read_bytes() != content:
                raise SystemExit(f"Artefacto desactualizado: {path.name}")
        else:
            path.write_bytes(content)


if __name__ == "__main__":
    main()
