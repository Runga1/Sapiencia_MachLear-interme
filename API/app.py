"""Sapiencia ML Intermedio - module reviewed and updated in 2026."""

import streamlit as st
import pandas as pd
from predictor import run_predictions
import tempfile
import os

# Título de la app
st.title("Cash Hunter")

# Instrucciones
st.write("Sube un archivo CSV con datos de transacciones para predecir fraudes.")

# Cargar el archivo CSV a través de un uploader
uploaded_file = st.file_uploader("Elige un archivo CSV", type=["csv"])

# Si el archivo es cargado
if uploaded_file is not None:
    try:
        # Mostrar el archivo cargado (primeras filas)
        df = pd.read_csv(uploaded_file)
        st.write("Vista previa de los datos:", df.head())

        # Crear un archivo temporal para procesar los datos
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_file.getvalue())  # Guardamos el contenido del archivo subido
            temp_filepath = tmp.name

        # Procesar la predicción usando el archivo temporal
        df_result = run_predictions(temp_filepath)

        # Limpiar el archivo temporal
        os.remove(temp_filepath)

        # Mostrar las predicciones
        if df_result is not None and not df_result.empty:
            st.write("Resultados de la predicción:")
            st.dataframe(df_result)

            # Descargar el archivo CSV con los resultados
            csv = df_result.to_csv(index=False)
            st.download_button(
                label="Descargar resultados",
                data=csv,
                file_name="predicciones.csv",
                mime="text/csv"
            )
        else:
            st.error("Error: No se pudo procesar el archivo")
    
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
