"""Añade la sección B2 al README original sin eliminar información previa.

Uso: python scripts/actualizar_readme.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SECCION = ROOT / "README_B2_AGREGAR.md"
MARCA = "## ✅ Entregable B2 — Build y Test: contrato de datos del dashboard"


def main():
    if not README.is_file() or not SECCION.is_file():
        raise SystemExit("Falta README.md original o README_B2_AGREGAR.md; integra los archivos en la raíz del repo.")
    actual = README.read_text(encoding="utf-8")
    if MARCA in actual:
        print("README.md ya contiene la sección B2. Actualiza allí las ligas reales cuando existan.")
        return
    texto = SECCION.read_text(encoding="utf-8")
    README.write_text(actual.rstrip() + "\n\n---\n\n" + texto, encoding="utf-8")
    print("README.md actualizado. Falta completar los enlaces REALES a corrida fallida y aprobada.")


if __name__ == "__main__":
    main()
