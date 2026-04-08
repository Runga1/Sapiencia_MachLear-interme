"""Sapiencia ML Intermedio - module reviewed and updated in 2026."""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import joblib

# === 1. Definir ruta de datos y carpeta para guardar modelo ===
DATA_PATH = 'C:/Users/Lenovo/.cache/kagglehub/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/versions/7/LI-Medium_Trans.csv'
MODEL_DIR = 'MODEL'

os.makedirs(MODEL_DIR, exist_ok=True)

print("Cargando datos...")
df = pd.read_csv(DATA_PATH)

# Preprocesamiento fechas
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day
df['Hour'] = df['Timestamp'].dt.hour
df.drop(columns=['Timestamp'], inplace=True)

# Función para mapear categorías raras a 'Other'
def replace_rare_categories(series, threshold=100):
    counts = series.value_counts()
    rare_cats = counts[counts < threshold].index
    return series.apply(lambda x: 'Other' if x in rare_cats else x)

# Aplicar reemplazo para columnas categóricas
for col in ['Receiving Currency', 'Payment Currency', 'Payment Format']:
    df[col] = df[col].astype(str)
    df[col] = replace_rare_categories(df[col], threshold=100)

# Limpiar datos
df = df.dropna(subset=['Is Laundering', 'Receiving Currency', 'Payment Currency', 'Payment Format'])
df = df.drop_duplicates()

print(f"Filas después limpieza y eliminación de duplicados: {len(df)}")

# Label encoding con 'Other' incluido explícitamente
label_encoders = {}
for col in ['Receiving Currency', 'Payment Currency', 'Payment Format']:
    le = LabelEncoder()
    le.fit(df[col])
    df[col] = le.transform(df[col])
    label_encoders[col] = le
    print(f"LabelEncoder para {col}: clases = {list(le.classes_)}")

# Variables predictoras y target
X = df.drop(columns=['Is Laundering', 'Account', 'Account.1'], errors='ignore')
y = df['Is Laundering']

print(f"Valores faltantes en y: {y.isna().sum()}")

# Separar train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# SMOTE para balancear clases
print("Aplicando SMOTE para balancear clases...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Calcular ratio para scale_pos_weight
ratio = float(np.sum(y_train_resampled == 0)) / np.sum(y_train_resampled == 1)
print(f"Ratio para scale_pos_weight: {ratio:.2f}")

# Entrenar modelo XGBoost
print("Entrenando modelo XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=ratio,
    eval_metric='logloss'
)
model.fit(X_train_resampled, y_train_resampled)

# Evaluar modelo
y_pred = model.predict(X_test)
print(f"Accuracy en test: {accuracy_score(y_test, y_pred):.4f}")
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Guardar modelo y encoders
joblib.dump(model, os.path.join(MODEL_DIR, 'xgb_SmallData.joblib'))
for col, le in label_encoders.items():
    joblib.dump(le, os.path.join(MODEL_DIR, f'label_encoder_{col}.joblib'))
print("Entrenamiento y guardado completados.")






