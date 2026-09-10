"""Inference Pipeline para el proyecto de precios de casas en Boston.

Carga el modelo y preprocessor entrenados por el Training Pipeline, lee datos
nuevos desde un archivo CSV, aplica las mismas transformaciones usadas
durante el entrenamiento, y genera predicciones de medv.

Nota de diseno: esta logica es funcionalmente equivalente a la funcion
predict() usada en la demo de Streamlit (notebooks/7-deploy/boston-streamlit.py).
Se mantienen separadas deliberadamente para no introducir riesgo sobre un
entregable ya cerrado del Trabajo 1 (ver PR de este Issue para el analisis
completo de la decision).

Ejecucion autonoma:
    python -m src.pipelines.inference_pipeline.inference_pipeline
"""

from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor

MODELS_DIR = Path("models")
NEW_DATA_PATH = Path("data/05_model_input/new_data.csv")
PREDICTIONS_OUTPUT_PATH = Path("data/07_model_output/predictions.csv")

RAW_FEATURE_COLUMNS = [
    "crim",
    "zn",
    "indus",
    "chas",
    "nox",
    "rm",
    "age",
    "dis",
    "tax",
    "ptratio",
    "lstat",
]


def load_artifacts(
    models_dir: Path,
) -> tuple[GradientBoostingRegressor, ColumnTransformer]:
    """Carga el modelo entrenado y el preprocessor ajustado.

    Args:
        models_dir: Ruta a la carpeta que contiene model.joblib y
            preprocessor.joblib.

    Returns:
        Tupla (modelo, preprocessor).
    """
    model = joblib.load(models_dir / "model.joblib")
    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    return model, preprocessor


def load_new_data(filepath: Path) -> pd.DataFrame:
    """Carga datos nuevos a predecir desde un archivo CSV.

    Args:
        filepath: Ruta al CSV con las columnas crudas del problema.

    Returns:
        DataFrame con los datos crudos, sin transformar.
    """
    df = pd.read_csv(filepath)
    df["chas"] = df["chas"].astype("boolean")
    return df


def predict(
    model: GradientBoostingRegressor,
    preprocessor: ColumnTransformer,
    df_raw: pd.DataFrame,
) -> np.ndarray:
    """Aplica las mismas transformaciones del entrenamiento y genera predicciones.

    Args:
        model: Modelo entrenado.
        preprocessor: Preprocessor ya ajustado (fit realizado en train_pipeline.py).
        df_raw: Datos nuevos, sin transformar.

    Returns:
        Array con las predicciones de medv (en miles de USD).
    """
    x_transformed = preprocessor.transform(df_raw[RAW_FEATURE_COLUMNS])
    return cast(np.ndarray, model.predict(x_transformed))


def save_predictions(df_raw: pd.DataFrame, predictions: np.ndarray, output_path: Path) -> None:
    """Guarda los datos originales junto con sus predicciones.

    Args:
        df_raw: Datos originales de entrada.
        predictions: Predicciones generadas.
        output_path: Ruta donde guardar el resultado.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resultado = df_raw.copy()
    resultado["medv_predicho"] = predictions
    resultado.to_csv(output_path, index=False)


def main() -> None:
    """Ejecuta el Inference Pipeline de forma autonoma."""
    model, preprocessor = load_artifacts(MODELS_DIR)
    df_raw = load_new_data(NEW_DATA_PATH)

    predictions = predict(model, preprocessor, df_raw)
    save_predictions(df_raw, predictions, PREDICTIONS_OUTPUT_PATH)

    print(f"Predicciones guardadas en {PREDICTIONS_OUTPUT_PATH}")
    print(f"Numero de predicciones generadas: {len(predictions)}")
    print(f"Rango de predicciones: {predictions.min():.2f} - {predictions.max():.2f}")


if __name__ == "__main__":
    main()
