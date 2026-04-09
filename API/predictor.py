"""Sapiencia ML Intermedio - module reviewed and updated in 2026."""

import pandas as pd
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Carpeta API

MODEL_PATH = os.path.join(BASE_DIR, "MODEL", "xgb_classifier_v2026.joblib")

ENCODERS_PATH = {
    'Receiving Currency': os.path.join(BASE_DIR, "MODEL", "label_encoder_Receiving Currency.joblib"),
    'Payment Currency': os.path.join(BASE_DIR, "MODEL", "label_encoder_Payment Currency.joblib"),
    'Payment Format': os.path.join(BASE_DIR, "MODEL", "label_encoder_Payment Format.joblib"),
}

# === Generador para leer datos por chunks ===
def data_generator(filepath, chunksize=10000):
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        yield chunk

# === Cargar modelo y encoders una sola vez ===
def load_model_and_encoders():
    model = joblib.load(MODEL_PATH)
    encoders = {col: joblib.load(path) for col, path in ENCODERS_PATH.items()}
    return model, encoders

# === Función segura para transformar con LabelEncoder manejando valores nuevos ===
def safe_transform(series, le):
    known_classes = set(le.classes_)
    most_freq_class = le.classes_[0]  # clase fallback
    # Reemplaza valores no conocidos con clase fallback
    series_fixed = series.apply(lambda x: x if x in known_classes else most_freq_class)
    return le.transform(series_fixed.astype(str))

# === Función para procesar un solo chunk ===
def predict_chunk(df, model, encoders):
    try:
        # Preprocesamiento temporal: fechas
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df['Year'] = df['Timestamp'].dt.year
            df['Month'] = df['Timestamp'].dt.month
            df['Day'] = df['Timestamp'].dt.day
            df['Hour'] = df['Timestamp'].dt.hour
            df.drop(columns=['Timestamp'], inplace=True)

        # Para columnas de moneda: reemplazar valores no conocidos con la clase más frecuente
        for col in ['Receiving Currency', 'Payment Currency']:
            if col in df.columns:
                known_currencies = set(encoders[col].classes_)
                most_freq_currency = encoders[col].classes_[0]
                df[col] = df[col].apply(lambda x: x if x in known_currencies else most_freq_currency)

        # Codificación con LabelEncoders usando safe_transform
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = safe_transform(df[col], le)
            else:
                raise ValueError(f"Falta la columna esperada: {col}")

        # Eliminar columnas no utilizadas
        for col in ['Is Laundering', 'Account.1']:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        # Preparar features para predicción
        df_features = df.drop(columns=['Account']) if 'Account' in df.columns else df.copy()

        # Alinear columnas al orden esperado por el modelo
        df_features = df_features[model.feature_names_in_]

        # Predecir probabilidades
        pred = model.predict_proba(df_features)[:, 1]

        # Preparar resultado
        result = pd.DataFrame({
            'Account': df['Account'] if 'Account' in df.columns else np.nan,
            'probability': pred
        })
        result['deadline'] = np.where(result['probability'] > 0.7, 'fraudulento', 'no_fraudulento')

        return result

    except Exception as e:
        print(f"Error en prediccion: {e}")
        return pd.DataFrame()

# === Función para correr todo el pipeline ===
def run_predictions(filepath, chunksize=10000):
    model, encoders = load_model_and_encoders()
    results = []

    for chunk in data_generator(filepath, chunksize):
        result_chunk = predict_chunk(chunk, model, encoders)
        if not result_chunk.empty:
            results.append(result_chunk)

    if results:
        final_result = pd.concat(results, ignore_index=True)
        return final_result
    else:
        return pd.DataFrame()
