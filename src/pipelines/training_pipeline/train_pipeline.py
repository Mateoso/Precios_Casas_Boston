"""Training Pipeline para el proyecto de precios de casas en Boston.

Lee las features procesadas por el Feature Pipeline, separa train/test,
ajusta el preprocessor (imputacion + escalado) solo con datos de
entrenamiento, entrena el modelo, evalua y guarda los artefactos
resultantes.

Ejecucion autonoma:
    python -m src.pipelines.training_pipeline.train_pipeline
"""

import warnings
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

FEATURES_PATH = Path("data/02_intermediate/boston_features.parquet")
PRIMARY_DATA_DIR = Path("data/03_primary")
MODELS_DIR = Path("models")
KEY_PREDICTORS = ["crim", "rm", "lstat"]
TARGET_COLUMN = "medv"
LOG_FEATURES = ["crim", "zn", "dis", "lstat"]
BOOLEAN_FEATURES = ["chas"]
DISTRIBUTION_THRESHOLD_STD = 0.5
MAX_NULL_DIFFERENCE = 0.05
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 10
CV_TEST_CONSISTENCY_THRESHOLD = 0.3
OVERFITTING_THRESHOLD = 0.6
MAE_BASELINE_HEURISTICO = 3.93

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


def validate_train_test_split(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Verifica que la separacion train/test sea correcta y representativa.

    Comprueba ausencia de overlap de indices (data leakage), similitud de
    distribucion del target entre ambos conjuntos, y proporcion comparable
    de nulos por columna.

    Args:
        x_train: Features de entrenamiento.
        x_test: Features de prueba.
        y_train: Target de entrenamiento.
        y_test: Target de prueba.

    Raises:
        ValueError: si se detecta overlap de indices entre train y test.

    Warns:
        UserWarning: si la distribucion del target o los nulos difieren
            mas alla del umbral esperado entre train y test.
    """
    overlap = set(x_train.index) & set(x_test.index)
    if overlap:
        raise ValueError(
            f"Se detectaron {len(overlap)} indices duplicados entre train y test: "
            "posible fuga de informacion (data leakage)."
        )

    y_completo = pd.concat([y_train, y_test])
    diferencia_medias = abs(y_train.mean() - y_test.mean())
    umbral = DISTRIBUTION_THRESHOLD_STD * y_completo.std()

    if diferencia_medias > umbral:
        warnings.warn(
            f"La diferencia de medias del target entre train ({y_train.mean():.2f}) "
            f"y test ({y_test.mean():.2f}) supera el umbral esperado "
            f"({umbral:.2f}). El split podria no ser representativo.",
            UserWarning,
            stacklevel=2,
        )
    for columna in KEY_PREDICTORS:
        diferencia_predictor = abs(x_train[columna].mean() - x_test[columna].mean())
        valores_completos = pd.concat([x_train[columna], x_test[columna]])
        umbral_predictor = DISTRIBUTION_THRESHOLD_STD * valores_completos.std()

        if diferencia_predictor > umbral_predictor:
            warnings.warn(
                f"La diferencia de medias de '{columna}' entre train "
                f"({x_train[columna].mean():.2f}) y test "
                f"({x_test[columna].mean():.2f}) supera el umbral esperado "
                f"({umbral_predictor:.2f}). El split podria no ser representativo.",
                UserWarning,
                stacklevel=2,
            )
    nulos_train = x_train.isna().mean()
    nulos_test = x_test.isna().mean()
    diferencia_nulos = (nulos_train - nulos_test).abs()

    if (diferencia_nulos > MAX_NULL_DIFFERENCE).any():
        columnas_afectadas = diferencia_nulos[diferencia_nulos > MAX_NULL_DIFFERENCE].index.tolist()
        warnings.warn(
            f"La proporcion de nulos difiere notablemente entre train y test "
            f"en las columnas: {columnas_afectadas}",
            UserWarning,
            stacklevel=2,
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


def validate_model(
    data_splits: dict[str, pd.DataFrame | pd.Series],
    model: GradientBoostingRegressor,
    preprocessor: ColumnTransformer,
) -> dict[str, dict[str, float] | str]:
    """Valida el modelo con cross-validation y compara train/CV/test.

    Ajusta un Pipeline (preprocessor + modelo) dentro de cada fold del KFold,
    evitando fuga de informacion. Genera dos diagnosticos independientes:
    (1) overfitting/underfitting, comparando train vs. test -- la comparacion
    clasica para este concepto; y (2) consistencia metodologica, comparando
    CV vs. test -- verifica que el set de test no este dando una estimacion
    optimista o pesimista por casualidad, ya que CV es una estimacion mas
    robusta del desempeno real al promediar sobre 10 particiones distintas.

    Args:
        x_train: Features de entrenamiento (sin transformar).
        y_train: Target de entrenamiento.
        x_test: Features de prueba (sin transformar).
        y_test: Target de prueba.
        model: Modelo a validar (sin entrenar).
        preprocessor: Preprocessor sin ajustar.

    Returns:
        Diccionario con metricas de train, cv (media y std) y test, mas un
        diagnostico textual de over/underfitting.
    """
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        full_pipeline,
        data_splits["x_train"],
        data_splits["y_train"],
        cv=kfold,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
    )

    full_pipeline.fit(data_splits["x_train"], data_splits["y_train"])
    pred_train = full_pipeline.predict(data_splits["x_train"])
    pred_test = full_pipeline.predict(data_splits["x_test"])

    metrics_train = {
        "MAE": mean_absolute_error(data_splits["y_train"], pred_train),
        "RMSE": mean_squared_error(data_splits["y_train"], pred_train) ** 0.5,
        "R2": r2_score(data_splits["y_train"], pred_train),
    }
    metrics_cv = {
        "MAE": -cv_results["test_neg_mean_absolute_error"].mean(),
        "MAE_std": cv_results["test_neg_mean_absolute_error"].std(),
        "RMSE": -cv_results["test_neg_root_mean_squared_error"].mean(),
        "R2": cv_results["test_r2"].mean(),
    }
    metrics_test = {
        "MAE": mean_absolute_error(data_splits["y_test"], pred_test),
        "RMSE": mean_squared_error(data_splits["y_test"], pred_test) ** 0.5,
        "R2": r2_score(data_splits["y_test"], pred_test),
    }

    brecha_relativa = (metrics_test["MAE"] - metrics_train["MAE"]) / metrics_train["MAE"]

    brecha_cv_test = (metrics_test["MAE"] - metrics_cv["MAE"]) / metrics_cv["MAE"]

    if abs(brecha_cv_test) <= CV_TEST_CONSISTENCY_THRESHOLD:
        consistencia = (
            f"Test y CV son consistentes entre si (diferencia relativa: "
            f"{brecha_cv_test:.1%}) -- el set de test no parece estar "
            "sobre ni sub-estimando el desempeno real del modelo."
        )
    else:
        consistencia = (
            f"ADVERTENCIA: Test y CV difieren notablemente (diferencia relativa: "
            f"{brecha_cv_test:.1%}) -- el set de test podria no ser representativo "
            "de la variabilidad real del problema. Considerar aumentar cv_folds "
            "o revisar el tamano del set de test."
        )
    if brecha_relativa > OVERFITTING_THRESHOLD:
        diagnostico = (
            f"OVERFITTING: la brecha relativa entre test y train "
            f"({brecha_relativa:.1%}) supera el umbral ({OVERFITTING_THRESHOLD:.0%}). "
            "Considerar mayor regularizacion (reducir max_depth, aumentar "
            "min_samples_leaf o subsample mas agresivo)."
        )
    elif metrics_train["MAE"] > MAE_BASELINE_HEURISTICO:
        diagnostico = (
            f"UNDERFITTING: el MAE de train ({metrics_train['MAE']:.2f}) supera "
            f"el del heuristico simple ({MAE_BASELINE_HEURISTICO:.2f}). "
            "Considerar mayor complejidad del modelo o mas features."
        )
    else:
        diagnostico = (
            f"ACEPTABLE: brecha relativa de {brecha_relativa:.1%} dentro del "
            f"umbral ({OVERFITTING_THRESHOLD:.0%}), y MAE de train supera al "
            "heuristico baseline."
        )

    return {
        "train": metrics_train,
        "cv": metrics_cv,
        "test": metrics_test,
        "diagnostico_overfitting": diagnostico,
        "diagnostico_consistencia": consistencia,
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
    df = load_features(FEATURES_PATH)
    x_train, x_test, y_train, y_test = split_data(df, TEST_SIZE, RANDOM_STATE)
    validate_train_test_split(x_train, x_test, y_train, y_test)

    preprocessor = build_preprocessor(x_train)
    model, fitted_preprocessor = train_model(x_train, y_train, preprocessor)

    metrics = evaluate_model(model, fitted_preprocessor, x_test, y_test)

    validacion_modelo = validate_model(
        {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test},
        GradientBoostingRegressor(**MODEL_PARAMS),
        build_preprocessor(x_train),
    )

    print(f"Validacion del modelo: {validacion_modelo}")

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
