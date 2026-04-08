"""Sapiencia ML Intermedio - module reviewed and updated in 2026."""

import joblib

encoder_paths = {
    'Receiving Currency': "API/MODEL/label_encoder_Receiving Currency.joblib",
    'Payment Currency': "API/MODEL/label_encoder_Payment Currency.joblib",
    'Payment Format': "API/MODEL/label_encoder_Payment Format.joblib"
}

for name, path in encoder_paths.items():
    le = joblib.load(path)
    print(f"Clases del encoder '{name}': {list(le.classes_)}")
    if 'Other' in le.classes_:
        print(f"✔️ 'Other' está incluido en '{name}'")
    else:
        print(f"❌ 'Other' NO está incluido en '{name}'")
