"""Run quick experiments to improve recall/overall performance.

Trains:
- RandomForest (baseline)
- RandomForest (class_weight='balanced')
- LogisticRegression (class_weight='balanced')

For each model, we evaluate at default threshold 0.5 and also search thresholds from 0.1 to 0.9
to show precision/recall/f1 trade-offs and pick a threshold that improves recall while
keeping reasonable precision.

Run: python -m src.experiments
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import joblib

from src.feature_engineering import preprocess


def eval_probs(y_true, probs, thresholds=[0.5]):
    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        rows.append({
            'threshold': t,
            'accuracy': accuracy_score(y_true, preds),
            'precision': precision_score(y_true, preds, zero_division=0),
            'recall': recall_score(y_true, preds, zero_division=0),
            'f1': f1_score(y_true, preds, zero_division=0)
        })
    return rows


def summarize(rows):
    for r in rows:
        print(f"t={r['threshold']:.2f} -> acc={r['accuracy']:.3f}, prec={r['precision']:.3f}, recall={r['recall']:.3f}, f1={r['f1']:.3f}")


def run():
    X, y, df = preprocess('data/sample_data.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    experiments = []

    # Baseline RF
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    p_rf = rf.predict_proba(X_test)[:, 1]
    experiments.append(('RF', rf, p_rf))

    # RF balanced
    rf_bal = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_bal.fit(X_train, y_train)
    p_rf_bal = rf_bal.predict_proba(X_test)[:, 1]
    experiments.append(('RF_balanced', rf_bal, p_rf_bal))

    # Logistic Regression balanced
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
    lr.fit(X_train, y_train)
    p_lr = lr.predict_proba(X_test)[:, 1]
    experiments.append(('Logistic_balanced', lr, p_lr))

    thresholds = np.linspace(0.1, 0.9, 17)

    for name, model, probs in experiments:
        print('\nModel:', name)
        print('Default threshold 0.5:')
        summarize(eval_probs(y_test, probs, thresholds=[0.5]))
        print('Threshold sweep:')
        rows = eval_probs(y_test, probs, thresholds=list(thresholds))
        # print top 3 by recall while keeping precision >= 0.2
        candidates = [r for r in rows if r['precision'] >= 0.2]
        candidates_sorted = sorted(candidates, key=lambda x: x['recall'], reverse=True)
        if candidates_sorted:
            print('Top candidates (precision>=0.2)')
            summarize(candidates_sorted[:3])
        else:
            print('No threshold found with precision>=0.2; showing full sweep:')
            summarize(rows[:5])

    # Save balanced RF as an alternative artifact
    artifact = {'model': rf_bal, 'feature_columns': list(X.columns)}
    joblib.dump(artifact, 'models/model_balanced.joblib')
    print('\nSaved balanced RF to models/model_balanced.joblib')


if __name__ == '__main__':
    run()
