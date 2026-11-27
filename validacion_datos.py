"""Contrato ejecutable del CSV del dashboard Netflix.

Esta revisión ocurre ANTES de generar KPIs, tablas, gráficas o predicciones.
La conversión es explícita y estricta: nunca se usa errors='coerce'
para ocultar un número inválido. Los campos numéricos quedan como dtypes reales.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import pandas as pd
from pandas.api.types import is_numeric_dtype

# Contrato del archivo fuente, no de las features de entrenamiento.
# (nombre, tipo, unidad, admite_ausente)
CAMPOS = {
    "customer_id": ("texto", "identificador", False),
    "age": ("entero", "años", False),
    "gender": ("texto", "categoría", True),
    "subscription_type": ("texto", "categoría", False),
    "watch_hours": ("decimal", "horas, ventana no documentada", False),
    "last_login_days": ("entero", "días", False),
    "region": ("texto", "categoría", True),
    "device": ("texto", "categoría", True),
    "monthly_fee": ("decimal", "unidad monetaria sin verificar", False),
    "churned": ("entero", "0=permanece; 1=cancela", False),
    "payment_method": ("texto", "categoría", True),
    "number_of_profiles": ("entero", "perfiles", False),
    "avg_watch_time_per_day": ("decimal", "horas/día", False),
    "favorite_genre": ("texto", "categoría", True),
}

# Política propuesta del equipo: ausencia/no capturado; NO equivale a 0.
MARCADORES_AUSENCIA = frozenset({"", "NA", "N/A", "NULL", "NONE", "SIN DATO", "--", "?"})
# Errores de fuente/planilla: hay que detener el tablero, NO imputar ni borrar.
MARCADORES_ERROR = frozenset({"ERROR", "ERR", "#N/A", "#VALUE!", "#DIV/0!", "INVÁLIDO", "INVALIDO"})


@dataclass
class ResultadoValidacion:
    datos: pd.DataFrame
    errores: list[str]
    advertencias: list[str]
    detalle: pd.DataFrame

    @property
    def valido(self) -> bool:
        return not self.errores


def _normalizar(serie: pd.Series) -> pd.Series:
    s = serie.astype("string").str.strip()
    return s.mask(s.str.upper().isin(MARCADORES_AUSENCIA).fillna(True), pd.NA)


def validar_dataset(df: pd.DataFrame) -> ResultadoValidacion:
    """Verifica esquema, tokens ausentes, valores, unicidad y dtypes.

    Siempre devuelve un reporte: Streamlit puede enseñar detalles de errores.
    Los tipos numéricos se convierten solo cuando TODOS los tokens son válidos.
    """
    errores: list[str] = []
    advertencias: list[str] = []
    detalle: list[dict] = []
    datos = df.copy()
    if datos.empty:
        errores.append("El CSV está vacío; no se puede calcular ningún indicador.")
    duplicadas_col = list(datos.columns[datos.columns.duplicated()])
    if duplicadas_col:
        errores.append(f"Nombres de columnas repetidos: {duplicadas_col}")
        return ResultadoValidacion(datos, errores, advertencias, pd.DataFrame())
    faltan = [col for col in CAMPOS if col not in datos.columns]
    if faltan:
        errores.append(f"Faltan columnas obligatorias: {', '.join(faltan)}")
    extras = [col for col in datos.columns if col not in CAMPOS and col != "segmento_inactividad"]
    if extras:
        advertencias.append(f"Columnas extra no publicadas ni auditadas: {', '.join(extras)}")

    for campo, (tipo, unidad, permite_ausencia) in CAMPOS.items():
        if campo not in datos.columns:
            detalle.append({"Columna": campo, "Tipo esperado": tipo, "Tipo encontrado": "AUSENTE", "Nulos": "—", "Estado": "ERROR"})
            continue
        original = datos[campo]
        if isinstance(original, pd.DataFrame):
            errores.append(f"{campo}: columna duplicada; corregir esquema.")
            continue
        serie = _normalizar(original)
        errores_marca = serie.str.upper().isin(MARCADORES_ERROR).fillna(False)
        if errores_marca.any():
            errores.append(f"{campo}: {int(errores_marca.sum())} marcador(es) explícitos de error; corregir la fuente.")
        nulos = int(serie.isna().sum())
        if nulos and not permite_ausencia:
            errores.append(f"{campo}: {nulos} ausencia(s) no permitidas; no se convierten en cero.")
        estado = "OK"
        if tipo in {"entero", "decimal"}:
            # dtype object/str no se acepta en la tabla final.
            convertido = pd.to_numeric(serie, errors="coerce")
            invalidos = serie.notna() & convertido.isna()
            if invalidos.any():
                muestras = serie.loc[invalidos].head(3).tolist()
                errores.append(f"{campo}: {int(invalidos.sum())} valor(es) no numéricos: {muestras}")
                estado = "ERROR"
            finitos = convertido.notna() & ~convertido.map(lambda x: math.isfinite(float(x)) if pd.notna(x) else True)
            if finitos.any():
                errores.append(f"{campo}: se detectaron infinitos o valores no finitos.")
                estado = "ERROR"
            if tipo == "entero":
                fracciones = convertido.notna() & (convertido % 1 != 0)
                if fracciones.any():
                    errores.append(f"{campo}: {int(fracciones.sum())} número(s) fraccionarios; se requiere entero.")
                    estado = "ERROR"
                if not invalidos.any() and not finitos.any() and not fracciones.any() and not nulos:
                    datos[campo] = convertido.astype("Int64" if permite_ausencia else "int64")
                else:
                    datos[campo] = convertido
            else:
                datos[campo] = convertido.astype(float)
            if not is_numeric_dtype(datos[campo].dtype):
                errores.append(f"{campo}: la conversión dejó una columna numérica como texto.")
                estado = "ERROR"
            # Se reportan tokens inválidos, no se 'arreglan' silenciosamente.
            if invalidos.any() or finitos.any():
                estado = "ERROR"
        else:
            datos[campo] = serie
        if nulos and not permite_ausencia or errores_marca.any():
            estado = "ERROR"
        detalle.append({"Columna": campo, "Tipo esperado": tipo, "Tipo encontrado": str(datos[campo].dtype), "Nulos": nulos, "Estado": estado})

    # Sólo probar restricciones cuando el campo ya quedó convertible.
    def rango(campo: str, minimo: float, maximo: float | None = None, estricto=False) -> None:
        if campo not in datos or not is_numeric_dtype(datos[campo].dtype):
            return
        s = datos[campo]
        falla = s.le(minimo) if estricto else s.lt(minimo)
        if maximo is not None:
            falla = falla | s.gt(maximo)
        if falla.fillna(False).any():
            errores.append(f"{campo}: {int(falla.fillna(False).sum())} valor(es) fuera del rango permitido.")

    rango("age", 0, 110)
    rango("watch_hours", 0)
    rango("last_login_days", 0)
    rango("monthly_fee", 0, estricto=True)
    rango("number_of_profiles", 1, 10)
    rango("avg_watch_time_per_day", 0, 24)
    if "churned" in datos and is_numeric_dtype(datos["churned"].dtype):
        no_binarios = datos["churned"].notna() & ~datos["churned"].isin([0, 1])
        if no_binarios.any():
            errores.append(f"churned: {int(no_binarios.sum())} valor(es) distintos de 0 o 1.")
    if "customer_id" in datos:
        duplicados = int(datos["customer_id"].dropna().duplicated().sum())
        if duplicados:
            errores.append(f"customer_id: {duplicados} identificador(es) repetidos.")
    if "segmento_inactividad" in datos and "last_login_days" in datos:
        # Es una columna calculada; nunca debe contradecir la columna fuente.
        if is_numeric_dtype(datos["last_login_days"].dtype):
            esperado = pd.cut(datos["last_login_days"], bins=[-float("inf"), 7, 30, float("inf")],
                              labels=["0-7 días", "8-30 días", "31+ días"]).astype("string")
            declarado = _normalizar(datos["segmento_inactividad"])
            desigual = (esperado != declarado).fillna(True)
            if desigual.any():
                errores.append(f"segmento_inactividad: {int(desigual.sum())} valor(es) contradicen last_login_days.")
    if errores:
        # Un resultado global rojo deja explícito que no debe dibujarse la app.
        detalle.append({"Columna": "CONTRATO COMPLETO", "Tipo esperado": "sin errores", "Tipo encontrado": "falló", "Nulos": "—", "Estado": "ERROR"})
    return ResultadoValidacion(datos=datos, errores=errores, advertencias=advertencias,
                               detalle=pd.DataFrame(detalle))


def leer_y_validar(ruta: str | Path) -> ResultadoValidacion:
    # dtype='string' permite detectar y manejar explícitamente TODOS los marcadores.
    df = pd.read_csv(ruta, dtype="string", keep_default_na=False, low_memory=False)
    return validar_dataset(df)


def resumen_segmentos(df: pd.DataFrame) -> pd.DataFrame:
    """La salida conserva tipos numéricos y no mezcla cifras fuente con cálculos."""
    base = (df.groupby("subscription_type", dropna=False, observed=True)
            .agg(Clientes=("churned", "size"), Abandonos=("churned", "sum"))
            .reset_index())
    base["Tasa de abandono (%)"] = 100.0 * base["Abandonos"] / base["Clientes"]
    return base.rename(columns={"subscription_type": "Plan"})
