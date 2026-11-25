"""Demostración académica: cuentas ficticias, secretos del host y permisos en servidor."""
import hmac
import streamlit as st

PAGINAS = {
    'gerencia': ['🏠 Resumen ejecutivo', '📊 Explorar clientes', '📣 Campañas de retención'],
    'analista': ['🏠 Resumen ejecutivo', '📊 Explorar clientes', '🤖 Comparar modelos',
                 '🎯 Predecir abandono', '📣 Campañas de retención'],
}


def iniciar_sesion():
    try:
        usuarios = st.secrets['usuarios']
    except (KeyError, FileNotFoundError):
        st.error('Falta configurar las cuentas de demostración en Secrets.')
        st.stop()
    usuario = st.session_state.get('usuario_autenticado')
    cuenta = usuarios.get(usuario) if usuario else None
    if cuenta is not None and cuenta.get('rol') in PAGINAS:
        rol = cuenta['rol']
        st.sidebar.caption(f'Sesión: {usuario} · Rol: {rol}')
        if st.sidebar.button('Cerrar sesión', key='cerrar_sesion'):
            st.session_state.clear()
            st.rerun()
        return rol
    st.session_state.pop('usuario_autenticado', None)
    st.title('Netflix · Acceso al dashboard')
    with st.form('acceso', clear_on_submit=True):
        nombre = st.text_input('Usuario')
        clave = st.text_input('Contraseña', type='password')
        entrar = st.form_submit_button('Iniciar sesión')
    if entrar:
        cuenta = usuarios.get(nombre.strip())
        if (cuenta is not None and cuenta.get('rol') in PAGINAS
                and isinstance(cuenta.get('password'), str)
                and bool(cuenta['password'])
                and hmac.compare_digest(clave.encode(), cuenta['password'].encode())):
            st.session_state.clear()
            st.session_state['usuario_autenticado'] = nombre.strip()
            st.rerun()
        st.error('Credenciales incorrectas.')
    st.stop()


def exigir_pagina(rol, pagina):
    if pagina not in PAGINAS.get(rol, []):
        st.error('No tienes permiso para acceder a esta sección.')
        st.stop()
