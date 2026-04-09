"""Sapiencia ML Intermedio - module reviewed and updated in 2026."""

import pandas as pd
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "MODEL", "xgb_classifier_v2026.joblib")

ENCODERS_PATH = {
    'Receiving Currency': os.path.join(BASE_DIR, "MODEL", "label_encoder_receiving_currency.joblib"),
    'Payment Currency': os.path.join(BASE_DIR, "MODEL", "label_encoder_payment_currency.joblib"),
    'Payment Format': os.path.join(BASE_DIR, "MODEL", "label_encoder_payment_format.joblib"),
}

def data_generator(filepath, chunksize=10000):
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        yield chunk

def load_model_and_encoders():
    model = joblib.load(MODEL_PATH)
    encoders = {col: joblib.load(path) for col, path in ENCODERS_PATH.items()}
    return model, encoders

def safe_transform(series, le):
    known_classes = set(le.classes_)
    most_freq_class = le.classes_[0]
    series_fixed = series.apply(lambda x: x if x in known_classes else most_freq_class)
    return le.transform(series_fixed.astype(str))

def predict_chunk(df, model, encoders):
    try:
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df['Year'] = df['Timestamp'].dt.year
            df['Month'] = df['Timestamp'].dt.month
            df['Day'] = df['Timestamp'].dt.day
            df['Hour'] = df['Timestamp'].dt.hour
            df.drop(columns=['Timestamp'], inplace=True)

        for col in ['Receiving Currency', 'Payment Currency']:
            if col in df.columns:
                known_currencies = set(encoders[col].classes_)
                most_freq_currency = encoders[col].classes_[0]
                df[col] = df[col].apply(lambda x: x if x in known_currencies else most_freq_currency)

        for col, le in encoders.items():
            if col in df.columns:
                df[col] = safe_transform(df[col], le)
            else:
                raise ValueError(f"Falta la columna esperada: {col}")

        for col in ['Is Laundering', 'Account.1']:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        df_features = df.drop(columns=['Account']) if 'Account' in df.columns else df.copy()
        df_features = df_features[model.feature_names_in_]

        pred = model.predict_proba(df_features)[:, 1]

        result = pd.DataFrame({
            'Account': df['Account'] if 'Account' in df.columns else np.nan,
            'probability': pred
        })
        result['deadline'] = np.where(result['probability'] > 0.7, 'fraudulento', 'no_fraudulento')

        return result

    except Exception as e:
        print(f"Error en prediccion: {e}")
        return pd.DataFrame()

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
    return pd.DataFrame()

    if results:
        final_result = pd.concat(results, ignore_index=True)
        return final_result
    else:
        return pd.DataFrame()
