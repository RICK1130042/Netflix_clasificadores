"""Pruebas deterministas; no hay que tocar el CSV real para probar los fallos."""
import sys
from pathlib import Path
import unittest

import pandas as pd
from pandas.api.types import is_numeric_dtype

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validacion_datos import CAMPOS, validar_dataset, resumen_segmentos  # noqa: E402


def caso_base():
    return pd.DataFrame([
        ["c-001", "31", "Female", "Basic", "14.73", "29", "Africa", "TV", "8.99", "1", "Card", "1", "0.49", "Drama"],
        ["c-002", "47", "Male", "Premium", "13.8", "0", "Europe", "Mobile", "17.99", "0", "PayPal", "3", "2.0", "Comedy"],
    ], columns=list(CAMPOS))


class ContratoDatosTests(unittest.TestCase):
    def test_valido_convierte_todas_las_columnas_numericas(self):
        reporte = validar_dataset(caso_base())
        self.assertTrue(reporte.valido, reporte.errores)
        for campo, (tipo, _, _) in CAMPOS.items():
            if tipo in {"entero", "decimal"}:
                self.assertTrue(is_numeric_dtype(reporte.datos[campo].dtype), campo)
        self.assertAlmostEqual(reporte.datos.loc[0, "monthly_fee"], 8.99)

    def test_numero_con_texto_bloquea(self):
        df = caso_base()
        df.loc[0, "monthly_fee"] = "8.99 ERROR"
        reporte = validar_dataset(df)
        self.assertFalse(reporte.valido)
        self.assertTrue(any("monthly_fee" in x for x in reporte.errores))

    def test_error_excel_bloquea(self):
        df = caso_base()
        df.loc[0, "monthly_fee"] = "#DIV/0!"
        self.assertFalse(validar_dataset(df).valido)

    def test_nulo_critico_no_se_convierte_a_cero(self):
        df = caso_base()
        df.loc[0, "watch_hours"] = "N/A"
        reporte = validar_dataset(df)
        self.assertFalse(reporte.valido)
        self.assertTrue(pd.isna(reporte.datos.loc[0, "watch_hours"]))

    def test_ausencia_opcional_es_nulo_no_cero(self):
        df = caso_base()
        df.loc[0, "favorite_genre"] = "Sin dato"
        reporte = validar_dataset(df)
        self.assertTrue(reporte.valido, reporte.errores)
        self.assertTrue(pd.isna(reporte.datos.loc[0, "favorite_genre"]))

    def test_cero_es_un_valor_verdadero(self):
        reporte = validar_dataset(caso_base())
        self.assertTrue(reporte.valido)
        self.assertEqual(reporte.datos.loc[1, "last_login_days"], 0)
        self.assertEqual(reporte.datos.loc[1, "churned"], 0)

    def test_churn_fuera_de_dominio_falla(self):
        df = caso_base()
        df.loc[0, "churned"] = "2"
        self.assertFalse(validar_dataset(df).valido)

    def test_entero_fraccionario_falla(self):
        df = caso_base()
        df.loc[0, "number_of_profiles"] = "1.5"
        self.assertFalse(validar_dataset(df).valido)

    def test_ids_repetidos_fallan(self):
        df = caso_base()
        df.loc[1, "customer_id"] = "c-001"
        self.assertFalse(validar_dataset(df).valido)

    def test_columna_faltante_falla(self):
        self.assertFalse(validar_dataset(caso_base().drop(columns="monthly_fee")).valido)

    def test_moneda_negativa_falla(self):
        df = caso_base()
        df.loc[0, "monthly_fee"] = "-5.25"
        self.assertFalse(validar_dataset(df).valido)

    def test_agregados_siguen_siendo_numericos(self):
        result = resumen_segmentos(validar_dataset(caso_base()).datos)
        self.assertTrue(is_numeric_dtype(result["Clientes"].dtype))
        self.assertTrue(is_numeric_dtype(result["Abandonos"].dtype))
        self.assertTrue(is_numeric_dtype(result["Tasa de abandono (%)"].dtype))
        self.assertAlmostEqual(result["Tasa de abandono (%)"].sum(), 100.0)

    def test_segmento_derivado_incorrecto_falla(self):
        df = caso_base()
        df["segmento_inactividad"] = ["0-7 días", "0-7 días"]
        self.assertFalse(validar_dataset(df).valido)


if __name__ == "__main__":
    unittest.main()
