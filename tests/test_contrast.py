import unittest
import tomllib
from pathlib import Path
from palette import COLORES, PARES, contraste

ROOT = Path(__file__).resolve().parents[1]


class TestContraste(unittest.TestCase):
    def test_wcag(self):
        for fg, bg, minimo in PARES:
            with self.subTest(fg=fg, bg=bg, minimo=minimo):
                ratio = contraste(COLORES[fg], COLORES[bg])
                print(f'{fg}/{bg}: {ratio:.3f}:1 (mínimo {minimo})')
                self.assertGreaterEqual(ratio, minimo)

    def test_formula(self):
        self.assertAlmostEqual(contraste('#000000', '#FFFFFF'), 21)
        self.assertAlmostEqual(contraste('#FFFFFF', '#FFFFFF'), 1)

    def test_tema(self):
        tema = tomllib.loads((ROOT / '.streamlit/config.toml').read_text())['theme']
        for clave, token in {'primaryColor': 'rojo', 'backgroundColor': 'fondo',
                            'secondaryBackgroundColor': 'tarjeta', 'textColor': 'texto'}.items():
            self.assertEqual(tema[clave], COLORES[token])
        self.assertEqual(tema['base'], 'dark')
