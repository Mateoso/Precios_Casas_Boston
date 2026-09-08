"""Training Pipeline para el proyecto de precios de casas en Boston.

Lee las features procesadas por el Feature Pipeline, separa train/test,
ajusta el preprocessor (imputacion + escalado) solo con datos de
entrenamiento, entrena el modelo, evalua y guarda los artefactos
resultantes.

Ejecucion autonoma:
    python -m src.pipelines.training_pipeline.train_pipeline
"""

from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

FEATURES_PATH = Path("data/02_intermediate/boston_features.parquet")
PRIMARY_DATA_DIR = Path("data/03_primary")
MODELS_DIR = Path("models")

TARGET_COLUMN = "medv"
LOG_FEATURES = ["crim", "zn", "dis", "lstat"]
BOOLEAN_FEATURES = ["chas"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_leaf": 5,
    "n_estimators": 150,
    "subsample": 0.7,
    "random_state": RANDOM_STATE,
}


def load_features(filepath: Path) -> pd.DataFrame:
    """Carga las features procesadas por el Feature Pipeline.

    Args:
        filepath: Ruta al archivo parquet de features.

    Returns:
        DataFrame con las features, sin imputar ni escalar.
    """
    return pd.read_parquet(filepath)


def split_data(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa el dataset en conjuntos de entrenamiento y prueba.

    Args:
        df: DataFrame de features completo, incluyendo el target.
        test_size: Proporcion del dataset para el conjunto de prueba.
        random_state: Semilla para reproducibilidad.

    Returns:
        Tupla (x_train, x_test, y_train, y_test).
    """
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(x, y, test_size=test_size, random_state=random_state),
    )


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Construye el preprocessor inspeccionando las columnas reales del DataFrame.

    Clasifica las columnas segun su naturaleza: log-transform para las de
    skewness fuerte (crim, zn, dis, lstat), escalado estandar para el resto
    de numericas, e imputacion por moda para la booleana (chas).

    Args:
        df: DataFrame de features (sin la columna target).

    Returns:
        ColumnTransformer sin ajustar.
    """
    log_features = [c for c in LOG_FEATURES if c in df.columns]
    boolean_features = [c for c in BOOLEAN_FEATURES if c in df.columns]
    numeric_features = [c for c in df.columns if c not in log_features + boolean_features]

    log_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    boolean_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("log", log_pipe, log_features),
            ("numeric", numeric_pipe, numeric_features),
            ("boolean", boolean_pipe, boolean_features),
        ]
    )


def train_model(
    x_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer
) -> tuple[GradientBoostingRegressor, ColumnTransformer]:
    """Ajusta el preprocessor con train y entrena el modelo.

    Args:
        x_train: Features de entrenamiento (sin transformar).
        y_train: Target de entrenamiento.
        preprocessor: ColumnTransformer sin ajustar.

    Returns:
        Tupla (modelo entrenado, preprocessor ajustado).
    """
    x_train_transformed = preprocessor.fit_transform(x_train)
    model = GradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(x_train_transformed, y_train)
    return model, preprocessor


def evaluate_model(
    model: GradientBoostingRegressor,
    preprocessor: ColumnTransformer,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evalua el modelo en el conjunto de prueba.

    Args:
        model: Modelo entrenado.
        preprocessor: Preprocessor ya ajustado con train.
        x_test: Features de prueba (sin transformar).
        y_test: Target de prueba.

    Returns:
        Diccionario con MAE, RMSE y R2.
    """
    x_test_transformed = preprocessor.transform(x_test)
    predictions = model.predict(x_test_transformed)
    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2": r2_score(y_test, predictions),
    }


def save_artifacts(
    model: GradientBoostingRegressor,
    preprocessor: ColumnTransformer,
    metrics: dict[str, float],
    data_splits: dict[str, pd.DataFrame | pd.Series],
) -> None:
    """Guarda el modelo, el preprocessor, las metricas y los splits de datos.

    Args:
        model: Modelo entrenado.
        preprocessor: Preprocessor ajustado.
        metrics: Diccionario de metricas de evaluacion.
        x_train: Features de entrenamiento sin transformar.
        x_test: Features de prueba sin transformar.
        y_train: Target de entrenamiento.
        y_test: Target de prueba.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PRIMARY_DATA_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODELS_DIR / "model.joblib")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    data_splits["x_train"].to_parquet(PRIMARY_DATA_DIR / "x_train.parquet", index=False)
    data_splits["x_test"].to_parquet(PRIMARY_DATA_DIR / "x_test.parquet", index=False)
    data_splits["y_train"].to_frame().to_parquet(PRIMARY_DATA_DIR / "y_train.parquet", index=False)
    data_splits["y_test"].to_frame().to_parquet(PRIMARY_DATA_DIR / "y_test.parquet", index=False)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(MODELS_DIR / "metrics.csv", index=False)


def main() -> None:
    """Ejecuta el Training Pipeline de forma autonoma."""
    df = load_features(FEATURES_PATH)
    x_train, x_test, y_train, y_test = split_data(df, TEST_SIZE, RANDOM_STATE)

    preprocessor = build_preprocessor(x_train)
    model, fitted_preprocessor = train_model(x_train, y_train, preprocessor)

    metrics = evaluate_model(model, fitted_preprocessor, x_test, y_test)
    save_artifacts(
        model,
        fitted_preprocessor,
        metrics,
        {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test},
    )

    print(f"Modelo y preprocessor guardados en {MODELS_DIR}")
    print(f"Metricas: {metrics}")


if __name__ == "__main__":
    main()
