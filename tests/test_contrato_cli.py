import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from test_validacion_datos import caso_base
from validacion_datos import validar_dataset

class TestContratoCLI(unittest.TestCase):
    def test_salida_roja_y_verde_sin_tocar_fuente(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'sintetico.csv'
            df = caso_base()
            for value, expected in [('8.99 ERROR', 1), ('8.99', 0)]:
                df.loc[0, 'monthly_fee'] = value
                df.to_csv(p, index=False)
                result = subprocess.run([sys.executable, 'scripts/validar_contrato.py', '--csv', str(p)], capture_output=True)
                self.assertEqual(result.returncode, expected, result.stdout)
    def test_ausencia_entera_reporta_error_sin_excepcion(self):
        for campo in ['age', 'last_login_days', 'number_of_profiles', 'churned']:
            with self.subTest(campo=campo):
                df = caso_base()
                df.loc[0, campo] = 'N/A'
                self.assertFalse(validar_dataset(df).valido)
    def test_columnas_duplicadas_no_provocan_excepcion(self):
        df = caso_base()
        df = df.loc[:, list(df.columns) + ['age']]
        self.assertFalse(validar_dataset(df).valido)
