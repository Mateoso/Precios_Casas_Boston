"""Pruebas unitarias para la validacion de datos del Feature Pipeline."""

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.pipelines.feature_pipeline.validate_data import (
    validate_features,
    validate_raw_data,
)


@pytest.fixture
def valid_raw_df() -> pd.DataFrame:
    """DataFrame crudo sintetico que cumple el schema de entrada."""
    return pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "crim": [0.05, 0.03, 0.02],
            "zn": [0.0, 12.5, 0.0],
            "indus": [8.0, 7.0, 6.0],
            "chas": [0.0, 1.0, 0.0],
            "nox": [0.5, 0.4, 0.45],
            "rm": [6.0, 6.5, 5.9],
            "age": [45.0, 30.0, 60.0],
            "dis": [4.0, 5.0, 3.5],
            "rad": [4.0, 3.0, 5.0],
            "tax": [300.0, 280.0, 310.0],
            "ptratio": [18.0, 17.0, 19.0],
            "black": [390.0, 395.0, 392.0],
            "lstat": [10.0, 8.0, 12.0],
            "medv": [24.0, 28.0, 22.0],
        }
    )


@pytest.fixture
def valid_features_df() -> pd.DataFrame:
    """DataFrame de features procesadas que cumple el schema de salida."""
    return pd.DataFrame(
        {
            "crim": [0.05, 0.03],
            "zn": [0.0, 12.5],
            "indus": [8.0, 7.0],
            "chas": pd.array([False, True], dtype="boolean"),
            "nox": [0.5, 0.4],
            "rm": [6.0, 6.5],
            "age": [45.0, 30.0],
            "dis": [4.0, 5.0],
            "tax": [300.0, 280.0],
            "ptratio": [18.0, 17.0],
            "lstat": [10.0, 8.0],
            "medv": [24.0, 28.0],
        }
    )


def test_validate_raw_data_acepta_datos_validos(valid_raw_df: pd.DataFrame) -> None:
    """Un DataFrame que cumple todas las reglas debe pasar sin error."""
    result = validate_raw_data(valid_raw_df)
    assert len(result) == len(valid_raw_df)


def test_validate_raw_data_rechaza_crim_negativo(valid_raw_df: pd.DataFrame) -> None:
    """Un valor de crim negativo debe hacer fallar la validacion."""
    invalid_df = valid_raw_df.copy()
    invalid_df.loc[0, "crim"] = -5.0
    with pytest.raises(SchemaErrors):
        validate_raw_data(invalid_df)


def test_validate_raw_data_rechaza_chas_fuera_de_rango(
    valid_raw_df: pd.DataFrame,
) -> None:
    """Un valor de chas distinto de 0/1 debe hacer fallar la validacion."""
    invalid_df = valid_raw_df.copy()
    invalid_df.loc[0, "chas"] = 5.0
    with pytest.raises(SchemaErrors):
        validate_raw_data(invalid_df)


def test_validate_raw_data_rechaza_exceso_de_nulos(valid_raw_df: pd.DataFrame) -> None:
    """Una columna con mas del 5% de nulos debe hacer fallar la validacion."""
    invalid_df = valid_raw_df.copy()
    invalid_df["crim"] = np.nan
    with pytest.raises(SchemaErrors):
        validate_raw_data(invalid_df)


def test_validate_features_acepta_datos_validos(
    valid_features_df: pd.DataFrame,
) -> None:
    """Un DataFrame de features que cumple el schema debe pasar sin error."""
    result = validate_features(valid_features_df)
    assert len(result) == len(valid_features_df)


def test_validate_features_rechaza_columna_no_esperada(
    valid_features_df: pd.DataFrame,
) -> None:
    """Una columna extra (ej. ID sin excluir) debe hacer fallar la validacion."""
    invalid_df = valid_features_df.copy()
    invalid_df["ID"] = [1, 2]
    with pytest.raises(SchemaErrors):
        validate_features(invalid_df)


def test_validate_features_rechaza_duplicados(valid_features_df: pd.DataFrame) -> None:
    """Filas duplicadas deben hacer fallar la validacion."""
    invalid_df = pd.concat([valid_features_df, valid_features_df.iloc[[0]]])
    with pytest.raises(SchemaErrors):
        validate_features(invalid_df)


def test_validate_features_rechaza_medv_nulo(valid_features_df: pd.DataFrame) -> None:
    """Un valor nulo en medv debe hacer fallar la validacion (nullable=False)."""
    invalid_df = valid_features_df.copy()
    invalid_df.loc[0, "medv"] = None
    with pytest.raises(SchemaErrors):
        validate_features(invalid_df)
