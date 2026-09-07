"""Pruebas unitarias para el Feature Pipeline."""

import pandas as pd
import pytest

from src.pipelines.feature_pipeline.feature_pipeline import (
    clean_data,
    fix_data_types,
    fix_scale_issues,
    select_features,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """DataFrame sintetico pequeno que imita la estructura del CSV crudo.

    Incluye deliberadamente: un valor de crim fuera de escala (5000), un
    duplicado exacto, y una fila sin valor en medv.
    """
    return pd.DataFrame(
        {
            "ID": [1, 2, 3, 3],
            "crim": [0.05, 5000.0, 0.03, 0.03],
            "zn": [0.0, 0.0, 12.5, 12.5],
            "indus": [8.0, 18.0, 7.0, 7.0],
            "chas": [0, 1, 0, 0],
            "nox": [0.5, 0.6, 0.4, 0.4],
            "rm": [6.0, 5.5, 6.5, 6.5],
            "age": [45.0, 80.0, 30.0, 30.0],
            "dis": [4.0, 2.0, 5.0, 5.0],
            "rad": [4, 24, 3, 3],
            "tax": [300, 666, 280, 280],
            "ptratio": [18.0, 20.0, 17.0, 17.0],
            "black": [390.0, 350.0, 395.0, 395.0],
            "lstat": [10.0, 22.0, 8.0, 8.0],
            "medv": [24.0, 18.0, None, None],
        }
    )


def test_fix_scale_issues_corrige_valores_fuera_de_rango(
    sample_raw_df: pd.DataFrame,
) -> None:
    """El valor de crim=5000 debe corregirse a 5.0 (dividido entre 1000)."""
    result = fix_scale_issues(sample_raw_df)
    assert result.loc[1, "crim"] == pytest.approx(5.0)


def test_fix_scale_issues_no_modifica_valores_ya_en_rango(
    sample_raw_df: pd.DataFrame,
) -> None:
    """Un valor de crim ya dentro de rango no debe alterarse."""
    result = fix_scale_issues(sample_raw_df)
    assert result.loc[0, "crim"] == pytest.approx(0.05)


def test_fix_data_types_asigna_boolean_a_chas(sample_raw_df: pd.DataFrame) -> None:
    """La columna chas debe quedar como dtype boolean, no int."""
    result = fix_data_types(sample_raw_df)
    assert str(result["chas"].dtype) == "boolean"


def test_fix_data_types_asigna_int64_a_rad(sample_raw_df: pd.DataFrame) -> None:
    """La columna rad debe quedar como dtype Int64 (nullable)."""
    result = fix_data_types(sample_raw_df)
    assert str(result["rad"].dtype) == "Int64"


def test_select_features_excluye_columnas_no_deseadas(
    sample_raw_df: pd.DataFrame,
) -> None:
    """ID, rad y black no deben estar en el resultado."""
    result = select_features(sample_raw_df)
    assert "ID" not in result.columns
    assert "rad" not in result.columns
    assert "black" not in result.columns


def test_select_features_conserva_columnas_esperadas(
    sample_raw_df: pd.DataFrame,
) -> None:
    """Las columnas predictoras y el target deben seguir presentes."""
    result = select_features(sample_raw_df)
    assert "crim" in result.columns
    assert "medv" in result.columns


def test_clean_data_elimina_duplicados(sample_raw_df: pd.DataFrame) -> None:
    """Las filas 2 y 3 son identicas y deben quedar reducidas a una sola."""
    df_sin_excluidas = select_features(sample_raw_df)
    result = clean_data(df_sin_excluidas)
    assert len(result) < len(df_sin_excluidas)


def test_clean_data_elimina_filas_sin_medv(sample_raw_df: pd.DataFrame) -> None:
    """Ninguna fila del resultado debe tener medv nulo."""
    df_sin_excluidas = select_features(sample_raw_df)
    result = clean_data(df_sin_excluidas)
    assert result["medv"].isna().sum() == 0
