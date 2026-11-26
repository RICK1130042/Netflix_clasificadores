"""Tokens de diseño utilizados por la app y las pruebas WCAG."""
COLORES = {
    'rojo': '#FF6B6B', 'fondo': '#111318', 'tarjeta': '#1B1F27',
    'borde': '#788394', 'texto': '#111318', 'texto_suave': '#B9BEC8',
    'verde': '#46D369', 'ambar': '#F5A524', 'gris': '#9AA0AA',
    'azul': '#56B4E9', 'negro': '#111318', 'blanco': '#FFFFFF',
}

# Cada par representa un uso real, no una comparación arbitraria de colores.
PARES = [(fg, bg, 4.5) for fg in ('texto', 'texto_suave', 'rojo', 'verde', 'ambar')
         for bg in ('fondo', 'tarjeta')]
PARES += [(fg, bg, 3.0) for fg in ('rojo', 'gris', 'azul', 'verde', 'ambar', 'borde', 'blanco')
          for bg in ('fondo', 'tarjeta')]
PARES += [('rojo', 'fondo', 3.0), ('texto', 'fondo', 3.0)]  # texto grande
PARES += [( 'negro', fg, 4.5) for fg in ('rojo', 'gris', 'azul', 'verde', 'ambar')]


def luminancia(hex_color):
    rgb = [int(hex_color[i:i+2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = [v / 12.92 if v <= 0.04045 else ((v + .055) / 1.055)**2.4 for v in rgb]
    return .2126*r + .7152*g + .0722*b


def contraste(a, b):
    l1, l2 = sorted((luminancia(a), luminancia(b)), reverse=True)
    return (l1 + .05) / (l2 + .05)


if __name__ == '__main__':
    for fg, bg, minimo in PARES:
        print(f'{fg:14} / {bg:8} {contraste(COLORES[fg], COLORES[bg]):.2f}:1 (mínimo {minimo})')
