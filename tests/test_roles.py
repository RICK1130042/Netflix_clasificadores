"""Prueba de auth.py real en Streamlit; secretos efímeros, sin datos del dashboard."""
import unittest
from streamlit.testing.v1 import AppTest
import auth

HARNESS = """
import streamlit as st
import auth
rol = auth.iniciar_sesion()
pagina = st.sidebar.radio('Navegación', auth.PAGINAS[rol])
auth.exigir_pagina(rol, pagina)
st.write('CONTENIDO AUTORIZADO: ' + pagina)
if st.session_state.get('forzar_pagina'):
    auth.exigir_pagina(rol, st.session_state['forzar_pagina'])
    st.write('PROTEGIDO')
"""

class TestRoles(unittest.TestCase):
    def app(self):
        at = AppTest.from_string(HARNESS)
        at.secrets['usuarios'] = {
            'test_g': {'rol': 'gerencia', 'password': 'solo-prueba-efimera'},
            'test_a': {'rol': 'analista', 'password': 'solo-prueba-efimera'},
            'test_x': {'rol': 'desconocido', 'password': 'solo-prueba-efimera'},
        }
        return at.run()
    def login(self, user, password='solo-prueba-efimera'):
        at = self.app()
        at.text_input[0].input(user)
        at.text_input[1].input(password)
        return at.button[0].click().run()
    def test_sin_sesion_no_hay_navegacion(self):
        at = self.app()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.radio), 0)
    def test_credenciales_invalidas_y_rol_desconocido(self):
        for user, password in [('test_g','incorrecta'), ('test_x','solo-prueba-efimera'), ('no_existe','x')]:
            at = self.login(user,password)
            self.assertEqual(len(at.radio), 0)
            self.assertTrue(at.error)
    def test_cada_rol_y_todas_sus_paginas(self):
        for user, rol in [('test_g','gerencia'), ('test_a','analista')]:
            at = self.login(user)
            self.assertFalse(at.exception)
            self.assertEqual(at.radio[0].options, auth.PAGINAS[rol])
            for pagina in auth.PAGINAS[rol]:
                at.radio[0].set_value(pagina).run()
                self.assertFalse(at.exception)
                self.assertTrue(any(pagina in m.value for m in at.markdown))
    def test_acceso_forzado_denegado(self):
        at = self.login('test_g')
        at.session_state['forzar_pagina'] = '🤖 Comparar modelos'
        at.run()
        self.assertTrue(at.error)
        self.assertFalse(any(m.value == 'PROTEGIDO' for m in at.markdown))
    def test_cerrar_sesion_y_cambiar_rol(self):
        at = self.login('test_a')
        at.button(key='cerrar_sesion').click().run()
        self.assertEqual(len(at.radio), 0)
        self.assertNotIn('usuario_autenticado', at.session_state)
        at.text_input[0].input('test_g')
        at.text_input[1].input('solo-prueba-efimera')
        at.button[0].click().run()
        self.assertEqual(at.radio[0].options, auth.PAGINAS['gerencia'])
