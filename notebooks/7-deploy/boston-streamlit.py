import os

import pandas as pd
import streamlit as st
from joblib import load

# https://docs.streamlit.io/library/api-reference

# HOW TO RUN THE APP:
# streamlit run notebooks/7-deploy/boston-streamlit.py

FEATURE_COLUMNS = [
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


def get_user_data() -> pd.DataFrame:
    """Get the data provided by the user for an individual prediction.

    Returns:
        DataFrame with a single row, matching the raw columns expected by
        the preprocessing pipeline.

    """
    user_data = {}

    col_a, col_b = st.columns(2)
    with col_a:
        user_data["crim"] = st.number_input(
            "Tasa de criminalidad (per cápita):",
            min_value=0.0,
            max_value=100.0,
            value=0.5,
            step=0.1,
        )
        user_data["zn"] = st.number_input(
            "% terreno residencial para lotes grandes:",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
        )
        user_data["indus"] = st.number_input(
            "% acres de negocios no minoristas:",
            min_value=0.0,
            max_value=30.0,
            value=8.0,
            step=0.5,
        )
        user_data["nox"] = st.slider(
            "Concentración de óxidos de nitrógeno:",
            min_value=0.3,
            max_value=0.9,
            value=0.5,
            step=0.01,
        )
        user_data["rm"] = st.slider(
            "Número promedio de cuartos:",
            min_value=3.0,
            max_value=9.0,
            value=6.0,
            step=0.1,
        )
        user_data["age"] = st.slider(
            "% viviendas construidas antes de 1940:",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
        )
    with col_b:
        user_data["dis"] = st.number_input(
            "Distancia ponderada a centros de empleo:",
            min_value=1.0,
            max_value=13.0,
            value=4.0,
            step=0.1,
        )
        user_data["tax"] = st.number_input(
            "Tasa de impuesto predial (por $10,000):",
            min_value=180,
            max_value=720,
            value=330,
            step=10,
        )
        user_data["ptratio"] = st.slider(
            "Ratio alumno-profesor:",
            min_value=12.0,
            max_value=22.0,
            value=18.0,
            step=0.1,
        )
        user_data["lstat"] = st.slider(
            "% población de estatus socioeconómico bajo:",
            min_value=1.0,
            max_value=38.0,
            value=12.0,
            step=0.5,
        )
        user_data["chas"] = st.radio(
            "¿Limita con el río Charles?:",
            options=["No", "Sí"],
            horizontal=True,
        )

    df = pd.DataFrame.from_dict(user_data, orient="index").T
    df["chas"] = df["chas"].map({"Sí": True, "No": False}).astype("boolean")

    return df[FEATURE_COLUMNS]


def preprocess_batch_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce types of batch CSV data before prediction.

    Args:
        df: Raw dataframe uploaded by the user.

    Returns:
        Dataframe with numeric/boolean types coerced, ready for the
        preprocessing pipeline.

    """
    processed_df = df.copy()

    if "chas" in processed_df.columns:
        if processed_df["chas"].dtype == "object":
            processed_df["chas"] = processed_df["chas"].map(
                lambda x: str(x).lower() in {"true", "1", "si", "sí", "yes"}
            )
        processed_df["chas"] = processed_df["chas"].astype("boolean")

    numeric_columns = [c for c in FEATURE_COLUMNS if c != "chas"]
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")

    return processed_df


@st.cache_resource
def load_artifacts(models_dir: str) -> tuple:
    """Load the preprocessing pipeline and the trained model.

    Args:
        models_dir: Path to the directory containing the .joblib files.

    Returns:
        Tuple of (preprocessor, model).

    """
    with st.spinner("Cargando modelo..."):
        preprocessor = load(os.path.join(models_dir, "preprocessor_pipeline.joblib"))
        model = load(os.path.join(models_dir, "best_model.joblib"))

    return preprocessor, model


def predict(preprocessor, model, df_raw: pd.DataFrame):
    """Apply the pipeline (preprocessor + model) to raw input data.

    Args:
        preprocessor: Fitted ColumnTransformer from Feature Engineering.
        model: Trained regression model.
        df_raw: Raw dataframe with the original feature columns.

    Returns:
        Array with predicted medv values (in thousands of USD).

    """
    x_transformed = preprocessor.transform(df_raw)
    feature_names = preprocessor.get_feature_names_out()
    x_transformed_df = pd.DataFrame(x_transformed, columns=feature_names)
    return model.predict(x_transformed_df)


def individual_prediction_tab(preprocessor, model) -> None:
    """Display the individual prediction interface."""
    df_user_data = get_user_data()

    prediction = predict(preprocessor, model, df_user_data)[0]

    st.write("")
    st.title(f"Valor estimado: ${prediction * 1000:,.0f} USD")
    st.metric("Precio estimado (medv)", f"{prediction:.2f} (miles de USD)")

    st.caption(
        "⚠️ Este modelo fue entrenado con datos de 1978 y tiene un techo de censura en "
        "$50,000 — las estimaciones cerca de ese valor pueden estar subestimadas."
    )


def batch_prediction_tab(preprocessor, model) -> None:
    """Display the batch prediction interface for CSV uploads."""
    st.subheader("Sube tu archivo CSV con datos de propiedades")

    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Vista previa de los datos cargados:")
            st.dataframe(df.head())

            missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]

            if missing_cols:
                st.warning(f"Faltan estas columnas: {', '.join(missing_cols)}")
                st.info(f"Columnas requeridas: {', '.join(FEATURE_COLUMNS)}")
            elif st.button("Predecir precios"):
                with st.spinner("Procesando y prediciendo..."):
                    df_processed = preprocess_batch_data(df)
                    predictions = predict(
                        preprocessor, model, df_processed[FEATURE_COLUMNS]
                    )

                    result_df = df.copy()
                    result_df["medv_predicho"] = predictions

                    st.success("¡Predicciones completadas!")
                    st.subheader("Resultados")
                    st.dataframe(result_df)

                    st.metric(
                        "Precio promedio predicho",
                        f"${predictions.mean() * 1000:,.0f} USD",
                    )

                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Descargar resultados como CSV",
                        data=csv,
                        file_name="boston_predictions.csv",
                        mime="text/csv",
                    )
        except (pd.errors.ParserError, ValueError, KeyError) as e:
            st.error(f"Error al procesar el archivo: {e}")
            st.info("Verifica que tu archivo CSV tenga el formato correcto.")
    else:
        st.info("Sube un archivo CSV con datos de propiedades.")
        st.subheader("Formato de ejemplo:")
        sample_data = pd.DataFrame(
            {
                "crim": [0.1, 5.5],
                "zn": [0.0, 0.0],
                "indus": [8.0, 18.0],
                "chas": [False, True],
                "nox": [0.5, 0.6],
                "rm": [6.2, 5.9],
                "age": [45, 80],
                "dis": [4.5, 2.1],
                "tax": [300, 400],
                "ptratio": [18.0, 20.0],
                "lstat": [10.0, 22.0],
            }
        )
        st.dataframe(sample_data)


def main() -> None:
    this_file_path = os.path.abspath(__file__)
    project_path = "/".join(this_file_path.split("/")[:-3])
    models_dir = os.path.join(project_path, "models")

    st.header("🏠 Predicción de Precios de Casas en Boston")
    st.caption(
        "Modelo Gradient Boosting (MAE test: 1.98 miles de USD) — "
        "predicciones basadas en 11 características de la vivienda y su entorno."
    )

    preprocessor, model = load_artifacts(models_dir)

    tab1, tab2 = st.tabs(["Predicción Individual", "Predicción por Lote (CSV)"])

    with tab1:
        individual_prediction_tab(preprocessor, model)

    with tab2:
        batch_prediction_tab(preprocessor, model)


if __name__ == "__main__":
    main()
