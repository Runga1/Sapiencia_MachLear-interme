import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path
from predictor import run_predictions
from db import init_db, get_session, Prediction

# Crear las tablas en la base de datos (una sola vez)
init_db()

# Crear dos columnas: título a la izquierda, imagen a la derecha
col1, col2 = st.columns([4, 1])

with col1:
    st.title("Cash Hunter")

with col2:
    image_path = Path(__file__).parent / "confusion_matrix_2026.png"
    if image_path.exists():
        st.image(str(image_path), width=80)
    else:
        st.warning("Imagen no encontrada.")

st.write("Sube un archivo CSV con las transacciones para obtener predicciones.")

uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df_preview = pd.read_csv(uploaded_file)
        st.write("Vista previa de los datos:")
        st.dataframe(df_preview.head())

        required_cols = ['Account', 'Receiving Currency', 'Payment Currency', 'Payment Format']
        missing_cols = [c for c in required_cols if c not in df_preview.columns]
        if missing_cols:
            st.error(f"Faltan columnas requeridas: {missing_cols}")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            df_result = run_predictions(tmp_path)
            os.remove(tmp_path)

            if df_result is not None and not df_result.empty:
                st.success("Predicciones realizadas correctamente")
                st.dataframe(df_result)

                session = get_session()
                try:
                    for _, row in df_result.iterrows():
                        pred = Prediction(
                            account=row['Account'],
                            probability=row['probability'],
                            deadline=row['deadline']
                        )
                        session.add(pred)
                    session.commit()
                    st.success("Resultados guardados en la base de datos.")
                except Exception as db_err:
                    session.rollback()
                    st.error(f"Error guardando en base de datos: {db_err}")
                finally:
                    session.close()

                csv = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Descargar resultados CSV",
                    data=csv,
                    file_name='predicciones.csv',
                    mime='text/csv'
                )
            else:
                st.error("No se pudo procesar el archivo o no hay resultados.")
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")




