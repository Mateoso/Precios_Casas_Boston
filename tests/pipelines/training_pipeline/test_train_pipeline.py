"""Pruebas unitarias para el Training Pipeline."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

import src.pipelines.training_pipeline.train_pipeline as tp
from src.pipelines.training_pipeline.train_pipeline import (
    build_preprocessor,
    evaluate_model,
    load_features,
    save_artifacts,
    split_data,
    train_model,
    validate_train_test_split,
)


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """DataFrame sintetico pequeno que imita la salida del Feature Pipeline."""
    n = 30
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        {
            "crim": rng.uniform(0.01, 10, n),
            "zn": rng.uniform(0, 50, n),
            "dis": rng.uniform(1, 10, n),
            "lstat": rng.uniform(1, 30, n),
            "indus": rng.uniform(1, 25, n),
            "nox": rng.uniform(0.4, 0.8, n),
            "rm": rng.uniform(4, 8, n),
            "age": rng.uniform(5, 95, n),
            "tax": rng.uniform(200, 700, n),
            "ptratio": rng.uniform(13, 21, n),
            "chas": pd.array(rng.choice([True, False], n), dtype="boolean"),
            "medv": rng.uniform(10, 45, n),
        }
    )


def test_load_features_carga_parquet_correctamente(tmp_path: Path) -> None:
    """load_features debe leer un parquet y devolver un DataFrame con las columnas esperadas."""
    n_rows_esperadas = 2
    sample_df = pd.DataFrame({"crim": [0.1, 0.2], "medv": [20.0, 25.0]})
    filepath = tmp_path / "test_features.parquet"
    sample_df.to_parquet(filepath, index=False)

    result = load_features(filepath)

    assert list(result.columns) == ["crim", "medv"]
    assert len(result) == n_rows_esperadas


def test_split_data_respeta_proporcion(sample_features_df: pd.DataFrame) -> None:
    """El split debe respetar aproximadamente el test_size solicitado."""
    n_test_esperado = 6
    n_train_esperado = 24
    x_train, x_test, _, _ = split_data(sample_features_df, test_size=0.2, random_state=42)
    assert len(x_test) == n_test_esperado
    assert len(x_train) == n_train_esperado


def test_split_data_no_comparte_indices(sample_features_df: pd.DataFrame) -> None:
    """Train y test no deben compartir ninguna fila (sin data leakage por overlap)."""
    x_train, x_test, _, _ = split_data(sample_features_df, test_size=0.2, random_state=42)
    overlap = set(x_train.index) & set(x_test.index)
    assert len(overlap) == 0


def test_build_preprocessor_clasifica_columnas_correctamente(
    sample_features_df: pd.DataFrame,
) -> None:
    """El preprocessor debe generar las columnas transformadas esperadas."""
    x = sample_features_df.drop(columns=["medv"])
    preprocessor = build_preprocessor(x)
    transformed = preprocessor.fit_transform(x)
    feature_names = preprocessor.get_feature_names_out()

    assert any("log__crim" in name for name in feature_names)
    assert any("numeric__indus" in name for name in feature_names)
    assert any("boolean__chas" in name for name in feature_names)
    assert transformed.shape[0] == len(x)


def test_train_model_produce_modelo_entrenado(
    sample_features_df: pd.DataFrame,
) -> None:
    """El modelo entrenado debe poder predecir sin lanzar error."""
    x = sample_features_df.drop(columns=["medv"])
    y = sample_features_df["medv"]
    preprocessor = build_preprocessor(x)

    model, fitted_preprocessor = train_model(x, y, preprocessor)

    assert isinstance(model, GradientBoostingRegressor)
    x_transformed = fitted_preprocessor.transform(x)
    predictions = model.predict(x_transformed)
    assert len(predictions) == len(x)


def test_evaluate_model_genera_metricas_esperadas(
    sample_features_df: pd.DataFrame,
) -> None:
    """Las metricas devueltas deben incluir MAE, RMSE y R2 con valores validos."""
    x_train, x_test, y_train, y_test = split_data(
        sample_features_df, test_size=0.3, random_state=42
    )
    preprocessor = build_preprocessor(x_train)
    model, fitted_preprocessor = train_model(x_train, y_train, preprocessor)

    metrics = evaluate_model(model, fitted_preprocessor, x_test, y_test)

    assert set(metrics.keys()) == {"MAE", "RMSE", "R2"}
    assert metrics["MAE"] >= 0
    assert metrics["RMSE"] >= 0


def test_save_artifacts_crea_los_archivos_esperados(
    sample_features_df: pd.DataFrame, tmp_path: Path
) -> None:
    """save_artifacts debe crear el modelo, preprocessor y metricas en disco."""
    original_models_dir = tp.MODELS_DIR
    original_primary_dir = tp.PRIMARY_DATA_DIR
    tp.MODELS_DIR = tmp_path / "models"
    tp.PRIMARY_DATA_DIR = tmp_path / "primary"

    try:
        x_train, x_test, y_train, y_test = split_data(
            sample_features_df, test_size=0.2, random_state=42
        )
        preprocessor = build_preprocessor(x_train)
        model, fitted_preprocessor = train_model(x_train, y_train, preprocessor)
        metrics = evaluate_model(model, fitted_preprocessor, x_test, y_test)

        save_artifacts(
            model,
            fitted_preprocessor,
            metrics,
            {
                "x_train": x_train,
                "x_test": x_test,
                "y_train": y_train,
                "y_test": y_test,
            },
        )

        assert (tp.MODELS_DIR / "model.joblib").exists()
        assert (tp.MODELS_DIR / "preprocessor.joblib").exists()
        assert (tp.MODELS_DIR / "metrics.csv").exists()
        assert (tp.PRIMARY_DATA_DIR / "x_train.parquet").exists()
    finally:
        tp.MODELS_DIR = original_models_dir
        tp.PRIMARY_DATA_DIR = original_primary_dir


def test_validate_train_test_split_detecta_overlap_de_indices(
    sample_features_df: pd.DataFrame,
) -> None:
    """Debe lanzar ValueError si train y test comparten indices."""
    x = sample_features_df.drop(columns=["medv"])
    y = sample_features_df["medv"]
    x_train_con_overlap = x.iloc[:20]
    x_test_con_overlap = x.iloc[15:25]
    y_train_con_overlap = y.iloc[:20]
    y_test_con_overlap = y.iloc[15:25]

    with pytest.raises(ValueError, match="indices duplicados"):
        validate_train_test_split(
            x_train_con_overlap,
            x_test_con_overlap,
            y_train_con_overlap,
            y_test_con_overlap,
        )


def test_validate_train_test_split_acepta_split_representativo(
    sample_features_df: pd.DataFrame,
) -> None:
    """No debe lanzar error ni warning con un split representativo."""
    x_train, x_test, y_train, y_test = split_data(
        sample_features_df, test_size=0.2, random_state=42
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_train_test_split(x_train, x_test, y_train, y_test)


def test_validate_train_test_split_advierte_distribucion_distinta() -> None:
    """Debe emitir UserWarning si las medias de train/test difieren demasiado."""
    x_train = pd.DataFrame({"crim": range(20)}, index=range(20))
    x_test = pd.DataFrame({"crim": range(20, 25)}, index=range(20, 25))
    y_train = pd.Series([10.0] * 20, index=range(20))
    y_test = pd.Series([45.0] * 5, index=range(20, 25))

    with pytest.warns(UserWarning, match="diferencia de medias"):
        validate_train_test_split(x_train, x_test, y_train, y_test)


def test_validate_train_test_split_advierte_nulos_distintos() -> None:
    """Debe emitir UserWarning si la proporcion de nulos difiere demasiado."""
    x_train = pd.DataFrame({"crim": [1.0] * 20}, index=range(20))
    x_test = pd.DataFrame({"crim": [np.nan] * 3 + [1.0] * 2}, index=range(20, 25))
    y_train = pd.Series([20.0] * 20, index=range(20))
    y_test = pd.Series([20.0] * 5, index=range(20, 25))

    with pytest.warns(UserWarning, match="proporcion de nulos"):
        validate_train_test_split(x_train, x_test, y_train, y_test)
