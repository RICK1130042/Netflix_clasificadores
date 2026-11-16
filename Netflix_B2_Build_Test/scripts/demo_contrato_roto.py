"""Demostración local segura: no toca el CSV original del repositorio."""
from pathlib import Path
import subprocess
import sys
import tempfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validacion_datos import CAMPOS  # noqa: E402


def main():
    fila = {
        "customer_id": "demo-01", "age": 31, "gender": "Female",
        "subscription_type": "Basic", "watch_hours": 10.5,
        "last_login_days": 15, "region": "Africa", "device": "TV",
        "monthly_fee": "8.99 ERROR", "churned": 1, "payment_method": "Card",
        "number_of_profiles": 2, "avg_watch_time_per_day": .65,
        "favorite_genre": "Drama",
    }
    with tempfile.TemporaryDirectory() as temporal:
        ruta = Path(temporal) / "datos_corruptos.csv"
        pd.DataFrame([fila], columns=list(CAMPOS)).to_csv(ruta, index=False)
        proceso = subprocess.run([sys.executable, str(ROOT / "scripts/validar_contrato.py"),
                                  "--csv", str(ruta)], capture_output=True, text=True)
        print(proceso.stdout)
        print(proceso.stderr)
        if proceso.returncode != 1 or "monthly_fee" not in proceso.stderr:
            raise SystemExit("Falló la demostración: el validador no detuvo el dato corrupto.")
        print("DEMO APROBADA: el validador SÍ rechazó el CSV corrupto con código 1.")


if __name__ == "__main__":
    main()
