"""Model training utilities.

Trains a RandomForestClassifier, evaluates on a test split, prints metrics and feature importances,
and saves the model along with the feature column list for consistent prediction.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import numpy as np
from typing import Tuple

from src.feature_engineering import preprocess


def train_model(data_path: str, model_out_path: str = 'models/model.joblib', random_state: int = 42, class_weight=None) -> Tuple[dict, dict]:
    """Train model and save it.

    Returns the trained model dict and evaluation dict.
    The saved object at `model_out_path` is a dict: {'model': model, 'feature_columns': [...]}.
    """
    X, y, df = preprocess(data_path)
    if y is None:
        raise ValueError('No target `no_show` found in data')

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )

    model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight=class_weight)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0))
    }

    # feature importances
    importances = dict(zip(X.columns, model.feature_importances_.tolist()))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]

    print('\nTraining completed. Evaluation metrics:')
    for k, v in metrics.items():
        print(f' - {k}: {v:.4f}')

    print('\nTop feature importances:')
    for name, imp in top_features:
        print(f' - {name}: {imp:.4f}')

    # persist model and feature columns
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    artifact = {'model': model, 'feature_columns': list(X.columns)}
    joblib.dump(artifact, model_out_path)
    print(f'Wrote model artifact to {model_out_path}')

    return artifact, {'metrics': metrics, 'top_features': top_features}


if __name__ == '__main__':
    # quick test when run directly
    from data.data_generator import generate_sample
    generate_sample(n=200)
    train_model('data/sample_data.csv')
