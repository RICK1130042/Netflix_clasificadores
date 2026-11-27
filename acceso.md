# Matriz de acceso y datos sensibles

| Vista o dato | Gerencia | Analista |
|---|---|---|
| Resumen, exploración agregada y panel gerencial de campañas | Sí | Sí |
| Modelos y predicción individual | — | Sí |
| CRM, diseño A/B y descargas individuales | — | Sí |
| Cargar CSV cuando no hay fuente local | — | Sí |

Las cuentas de demostración se provisionan en Streamlit Secrets. El rol se comprueba en el servidor por medio de `auth.py`. Las credenciales se envían al evaluador por un canal privado. En la demo, los identificadores de cliente, sus registros de actividad, forma de pago y puntuación individual son datos de acceso restringido dentro del tablero. El repositorio es público y el login no controla el acceso a los archivos versionados. Utilizar únicamente datos académicos autorizados.

Las [pruebas de roles](tests/test_roles.py) cubren ingreso, navegación autorizada, denegación, cierre y cambio de rol. Para la defensa, abrir la URL en ventana privada y demostrar ambas cuentas en sesiones separadas. El login de la app es distinto al inicio de sesión de Streamlit Cloud.
