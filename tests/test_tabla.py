import ast
import unittest
from pathlib import Path

class TestTabla(unittest.TestCase):
    def test_tabla_segmentos_declara_orden_y_oculta_indice(self):
        source = Path('app.py').read_text()
        tree = ast.parse(source)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == 'dataframe' and n.args and 'tabla_seg' in ast.unparse(n.args[0])]
        self.assertEqual(len(calls), 1)
        kw = {k.arg: k.value for k in calls[0].keywords}
        self.assertTrue(ast.literal_eval(kw['hide_index']))
        self.assertEqual(ast.literal_eval(kw['column_order']), ['Segmento','Clientes','Cancelan','Abandono (%)'])
        self.assertIn('CALCULADO · fuente: CSV activo del dashboard', source)
    def test_navegacion_protegida_antes_del_contenido(self):
        source = Path('app.py').read_text()
        self.assertLess(source.index('rol = auth.iniciar_sesion()'), source.index('pagina = st.sidebar.radio'))
        self.assertLess(source.index('auth.exigir_pagina(rol, pagina)'), source.index('if pagina =='))
        self.assertIn('if rol != "analista":', source)
