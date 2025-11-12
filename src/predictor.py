"""Prediction utilities: load model artifact, prepare features, predict risk scores and labels."""
from typing import Optional
import joblib
import pandas as pd
import os

from src.feature_engineering import preprocess


def load_artifact(path='models/model.joblib') -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model artifact not found: {path}')
    return joblib.load(path)


def predict_for_df(df_or_path, model_artifact: Optional[dict] = None, threshold: float = 0.5):
    """Return a DataFrame with columns: appointment_id, predicted_status, risk_score

    The model_artifact should be a dict with keys 'model' and 'feature_columns'. If not supplied,
    it will be loaded from `models/model.joblib`.
    """
    if model_artifact is None:
        model_artifact = load_artifact()

    model = model_artifact['model']
    feature_columns = model_artifact['feature_columns']

    X, y, processed = preprocess(df_or_path)

    # align columns
    X = X.reindex(columns=feature_columns, fill_value=0)

    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    out = pd.DataFrame({
        'appointment_id': processed.get('appointment_id', pd.Series([None]*len(X))).values,
        'predicted_status': ['No-show' if p==1 else 'Show' for p in preds],
        'risk_score': proba
    })

    return out


if __name__ == '__main__':
    # smoke test
    from data.data_generator import generate_sample
    generate_sample(n=100)
    art = load_artifact() if os.path.exists('models/model.joblib') else None
    if art is None:
        print('No artifact found; run training first (src.model_training.train_model)')
    else:
        res = predict_for_df('data/sample_data.csv', art)
        print(res.head())
