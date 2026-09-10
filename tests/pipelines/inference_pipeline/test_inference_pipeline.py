"""Pruebas unitarias para el Inference Pipeline, con modelo dummy y datos sinteticos."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler

from src.pipelines.inference_pipeline.inference_pipeline import (
    load_artifacts,
    load_new_data,
    predict,
    save_predictions,
)


@pytest.fixture
def dummy_model() -> DummyRegressor:
    """Modelo dummy que siempre predice el mismo valor fijo."""
    modelo = DummyRegressor(strategy="constant", constant=20.0)
    x_fake = pd.DataFrame({"crim": [0.1, 0.2], "rm": [6.0, 6.5]})
    y_fake = pd.Series([20.0, 20.0])
    modelo.fit(x_fake, y_fake)
    return modelo


@pytest.fixture
def dummy_preprocessor() -> ColumnTransformer:
    """Preprocessor simple ajustado sobre datos sinteticos, para pruebas aisladas."""
    x_fake = pd.DataFrame(
        {
            "crim": [0.1, 0.2],
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
        }
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                [
                    "crim",
                    "zn",
                    "indus",
                    "nox",
                    "rm",
                    "age",
                    "dis",
                    "tax",
                    "ptratio",
                    "lstat",
                ],
            ),
            ("bool", "passthrough", ["chas"]),
        ]
    )
    preprocessor.fit(x_fake)
    return preprocessor


@pytest.fixture
def sample_new_data() -> pd.DataFrame:
    """Datos sinteticos nuevos a predecir."""
    return pd.DataFrame(
        {
            "crim": [0.1, 0.2],
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
        }
    )


def test_load_artifacts_carga_modelo_y_preprocessor(
    tmp_path: Path, dummy_model: DummyRegressor, dummy_preprocessor: ColumnTransformer
) -> None:
    """load_artifacts debe cargar correctamente el modelo y preprocessor guardados."""
    joblib.dump(dummy_model, tmp_path / "model.joblib")
    joblib.dump(dummy_preprocessor, tmp_path / "preprocessor.joblib")

    modelo_cargado, preprocessor_cargado = load_artifacts(tmp_path)

    assert isinstance(modelo_cargado, DummyRegressor)
    assert isinstance(preprocessor_cargado, ColumnTransformer)


def test_load_new_data_convierte_chas_a_boolean(tmp_path: Path) -> None:
    """load_new_data debe convertir chas a dtype boolean, sin importar como llega en el CSV."""
    csv_path = tmp_path / "new_data.csv"
    pd.DataFrame(
        {
            "crim": [0.1],
            "zn": [0.0],
            "indus": [8.0],
            "chas": [False],
            "nox": [0.5],
            "rm": [6.0],
            "age": [45.0],
            "dis": [4.0],
            "tax": [300.0],
            "ptratio": [18.0],
            "lstat": [10.0],
        }
    ).to_csv(csv_path, index=False)

    result = load_new_data(csv_path)

    assert str(result["chas"].dtype) == "boolean"


def test_predict_genera_predicciones_con_modelo_dummy(
    dummy_model: DummyRegressor,
    dummy_preprocessor: ColumnTransformer,
    sample_new_data: pd.DataFrame,
) -> None:
    """Con un modelo dummy constante, todas las predicciones deben ser el valor fijo."""
    predicciones_esperadas = 20.0
    predictions = predict(dummy_model, dummy_preprocessor, sample_new_data)

    assert len(predictions) == len(sample_new_data)
    assert all(pred == predicciones_esperadas for pred in predictions)


def test_save_predictions_agrega_columna_medv_predicho(
    tmp_path: Path, sample_new_data: pd.DataFrame
) -> None:
    """save_predictions debe guardar un CSV con la columna medv_predicho agregada."""
    predictions = np.array([20.0, 20.0])
    output_path = tmp_path / "predictions.csv"

    save_predictions(sample_new_data, predictions, output_path)

    resultado = pd.read_csv(output_path)
    assert "medv_predicho" in resultado.columns
    assert list(resultado["medv_predicho"]) == list(predictions)
