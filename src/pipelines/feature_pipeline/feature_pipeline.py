"""Feature Pipeline para el proyecto de precios de casas en Boston.

Lee los datos crudos, aplica transformaciones deterministas (tipificación,
corrección de escala, selección de atributos y limpieza), y guarda el
resultado para ser consumido por el Training Pipeline.

No incluye imputación de nulos ni escalado (StandardScaler), ya que esas
transformaciones requieren ajuste (fit) sobre los datos y deben calcularse
después del split train/test, dentro del Training Pipeline, para evitar
fuga de información (data leakage).

Ejecución autónoma:
    python src/pipelines/feature_pipeline/feature_pipeline.py
"""

from pathlib import Path

import pandas as pd

from src.pipelines.feature_pipeline.validate_data import (
    validate_features,
    validate_raw_data,
)

RAW_DATA_PATH = Path("data/01_raw/Precios_Casas_Boston.csv")
OUTPUT_PATH = Path("data/02_intermediate/boston_features.parquet")

EXCLUDED_COLUMNS = ["ID", "rad", "black"]

CRIM_SCALE_THRESHOLD = 100
NOX_SCALE_THRESHOLD = 1
RM_SCALE_THRESHOLD = 15
DIS_SCALE_THRESHOLD = 20
SCALE_FACTOR = 1000


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Carga el dataset crudo de precios de casas en Boston.

    Args:
        filepath: Ruta al archivo CSV crudo.

    Returns:
        DataFrame con los datos originales, sin ninguna transformación.
    """
    return pd.read_csv(filepath)


def fix_scale_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige el error de escala (division entre 1000) en crim, nox, rm y dis.

    Args:
        df: DataFrame con las columnas originales.

    Returns:
        DataFrame con los valores de escala corregidos.
    """
    df = df.copy()
    df.loc[df["crim"] > CRIM_SCALE_THRESHOLD, "crim"] /= SCALE_FACTOR
    df.loc[df["nox"] > NOX_SCALE_THRESHOLD, "nox"] /= SCALE_FACTOR
    df.loc[df["rm"] > RM_SCALE_THRESHOLD, "rm"] /= SCALE_FACTOR
    df.loc[df["dis"] > DIS_SCALE_THRESHOLD, "dis"] /= SCALE_FACTOR
    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Tipifica las columnas segun su naturaleza semantica.

    Args:
        df: DataFrame con columnas en su tipo inferido por defecto.

    Returns:
        DataFrame con chas como booleano, rad como entero categorico ordinal,
        y el resto de columnas numericas como float64.
    """
    df = df.copy()
    df["chas"] = df["chas"].astype("boolean")
    df["rad"] = df["rad"].astype("Int64")

    numeric_columns = [
        "crim",
        "zn",
        "indus",
        "nox",
        "rm",
        "age",
        "dis",
        "tax",
        "ptratio",
        "black",
        "lstat",
        "medv",
    ]
    df[numeric_columns] = df[numeric_columns].astype("float64")
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Excluye columnas sin valor predictivo o redundantes.

    Se excluyen ID (no es predictiva ni es un identificador unico), rad
    (redundante con tax, correspondencia 1:1 confirmada con VIF) y black
    (construccion historica cuestionable y relacion estadisticamente
    inestable con el target, confirmada en el analisis bivariable).

    Args:
        df: DataFrame con todas las columnas originales.

    Returns:
        DataFrame solo con las columnas seleccionadas para el modelo.
    """
    return df.drop(columns=EXCLUDED_COLUMNS)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas duplicadas y filas sin valor en la variable objetivo.

    Args:
        df: DataFrame con las columnas ya seleccionadas.

    Returns:
        DataFrame limpio, sin duplicados ni nulos en medv, con indice
        reiniciado.
    """
    df = df.drop_duplicates()
    df = df.dropna(subset=["medv"])
    return df.reset_index(drop=True)


def build_features(raw_filepath: Path) -> pd.DataFrame:
    """Orquesta el pipeline completo de features deterministas.

    Args:
        raw_filepath: Ruta al CSV crudo de entrada.

    Returns:
        DataFrame limpio, tipificado y con escala corregida, listo para el
        Training Pipeline. No incluye imputacion ni escalado.
    """
    df = load_raw_data(raw_filepath)
    df = validate_raw_data(df)
    df = fix_scale_issues(df)
    df = fix_data_types(df)
    df = select_features(df)
    df = clean_data(df)
    df = validate_features(df)
    return df


def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """Guarda el dataset de features en formato parquet.

    Args:
        df: DataFrame de features procesadas.
        output_path: Ruta donde se guardara el archivo.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def main() -> None:
    """Ejecuta el Feature Pipeline de forma autonoma."""
    features_df = build_features(RAW_DATA_PATH)
    save_features(features_df, OUTPUT_PATH)
    print(f"Features guardadas en {OUTPUT_PATH}")
    print(f"Shape final: {features_df.shape}")


if __name__ == "__main__":
    main()
