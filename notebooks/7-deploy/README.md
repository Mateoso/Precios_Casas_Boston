# Demo — Predicción de Precios de Casas en Boston

Aplicación interactiva construida con [Streamlit](https://streamlit.io/) que permite predecir
el precio de una vivienda en Boston usando el modelo Gradient Boosting entrenado en el Paso 6
del proyecto (MAE test: 1.98 miles de USD).

## Cómo ejecutar

Desde la raíz del proyecto:

```bash
uv sync
uv run streamlit run notebooks/7-deploy/boston-streamlit.py
```

La aplicación se abrirá automáticamente en el navegador en `http://localhost:8501`. Si no se
abre sola (común en entornos WSL2), copia esa URL manualmente en tu navegador.

Para detener el servidor: `Ctrl+C` en la terminal.

## Requisitos previos

Deben existir los siguientes artefactos en la carpeta `models/` (generados en pasos
anteriores del proyecto):

- `preprocessor_pipeline.joblib` — pipeline de Feature Engineering (Paso 4)
- `best_model.joblib` — modelo Gradient Boosting entrenado (Paso 6)

Si no existen, deben regenerarse corriendo en orden los notebooks de los Pasos 1 a 6.

## Funcionalidades

### Predicción Individual

Formulario con las 11 variables predictoras del modelo (crim, zn, indus, chas, nox, rm, age,
dis, tax, ptratio, lstat), con valores por defecto razonables y rangos acotados a los
observados en el dataset original.

### Predicción por Lote (CSV)

Permite subir un archivo CSV con múltiples propiedades y obtener predicciones para todas a la
vez, con opción de descargar los resultados. El CSV debe incluir las 11 columnas requeridas
(ver formato de ejemplo dentro de la app si no se sube ningún archivo).

## Advertencia sobre el modelo

El dataset original (Boston Housing, 1978) tiene un techo de censura en $50,000 — las
propiedades con valor real igual o superior a ese monto fueron registradas como exactamente
$50,000. Las predicciones cercanas a ese valor pueden estar subestimadas (ver análisis
detallado en `notebooks/6-interpretation/`).
