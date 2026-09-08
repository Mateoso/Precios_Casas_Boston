"""Validacion de datos para el Feature Pipeline, usando Pandera.

Define dos schemas: uno para los datos crudos de entrada, y otro para las
features procesadas de salida. Si una validacion falla, se lanza una
excepcion explicita y no se persisten los features.
"""

from typing import cast

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

MAX_NULL_PERCENTAGE = 0.05

CRIM_SCALE_THRESHOLD = 100
NOX_SCALE_THRESHOLD = 1
RM_SCALE_THRESHOLD = 15
DIS_SCALE_THRESHOLD = 20

EXCLUDED_COLUMNS = ["ID", "rad", "black"]


def _check_max_null_percentage(df: pa.typing.DataFrame) -> bool:
    """Verifica que ninguna columna supere el porcentaje maximo de nulos.

    Args:
        df: DataFrame a validar.

    Returns:
        True si todas las columnas cumplen el umbral, False en caso contrario.
    """
    null_percentages = df.isna().mean()
    return bool((null_percentages <= MAX_NULL_PERCENTAGE).all())


raw_data_schema = DataFrameSchema(
    columns={
        "ID": Column(int, required=True),  # unico entero real, sin nulos
        "crim": Column(float, Check.ge(0.0), nullable=True),
        "zn": Column(float, Check.ge(0.0), nullable=True),
        "indus": Column(float, Check.ge(0.0), nullable=True),
        "chas": Column(float, Check.isin([0.0, 1.0]), nullable=True),
        "nox": Column(float, Check.ge(0.0), nullable=True),
        "rm": Column(float, Check.gt(0.0), nullable=True),
        "age": Column(float, Check.in_range(0.0, 100.0), nullable=True),
        "dis": Column(float, Check.gt(0.0), nullable=True),
        "rad": Column(float, nullable=True),
        "tax": Column(float, Check.gt(0.0), nullable=True),
        "ptratio": Column(float, Check.gt(0.0), nullable=True),
        "black": Column(float, nullable=True),
        "lstat": Column(float, Check.ge(0.0), nullable=True),
        "medv": Column(float, Check.gt(0.0), nullable=True),
    },
    checks=Check(
        _check_max_null_percentage,
        error=f"Alguna columna supera el {MAX_NULL_PERCENTAGE:.0%} maximo de nulos permitido",
    ),
    strict=False,
)


features_output_schema = DataFrameSchema(
    columns={
        "crim": Column(float, Check.lt(CRIM_SCALE_THRESHOLD), nullable=True),
        "zn": Column(float, Check.ge(0.0), nullable=True),
        "indus": Column(float, Check.ge(0.0), nullable=True),
        "chas": Column("boolean", nullable=True),
        "nox": Column(float, Check.le(NOX_SCALE_THRESHOLD), nullable=True),
        "rm": Column(float, Check.lt(RM_SCALE_THRESHOLD), nullable=True),
        "age": Column(float, Check.in_range(0.0, 100.0), nullable=True),
        "dis": Column(float, Check.lt(DIS_SCALE_THRESHOLD), nullable=True),
        "tax": Column(float, Check.gt(0.0), nullable=True),
        "ptratio": Column(float, Check.gt(0.0), nullable=True),
        "lstat": Column(float, Check.ge(0.0), nullable=True),
        "medv": Column(float, Check.gt(0.0), nullable=False),
    },
    checks=Check(
        lambda df: not df.duplicated().any(),
        error="El dataset de features no debe contener filas duplicadas",
    ),
    strict=True,
)


def validate_raw_data(df: pa.typing.DataFrame) -> pa.typing.DataFrame:
    """Valida el DataFrame crudo contra el schema de entrada.

    Args:
        df: DataFrame recien cargado desde el CSV crudo.

    Returns:
        El mismo DataFrame, si pasa la validacion.

    Raises:
        pandera.errors.SchemaErrors: si el DataFrame no cumple el schema.
    """
    return cast(pa.typing.DataFrame, raw_data_schema.validate(df, lazy=True))


def validate_features(df: pa.typing.DataFrame) -> pa.typing.DataFrame:
    """Valida las features procesadas contra el schema de salida.

    Args:
        df: DataFrame de features ya transformadas.

    Returns:
        El mismo DataFrame, si pasa la validacion.

    Raises:
        pandera.errors.SchemaErrors: si el DataFrame no cumple el schema.
    """
    return cast(pa.typing.DataFrame, features_output_schema.validate(df, lazy=True))
