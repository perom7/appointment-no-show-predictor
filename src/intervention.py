"""Intervention recommendation logic.

Maps risk scores to actions and writes an interventions CSV containing appointment_id,
predicted_status, risk_score, recommended_action.
"""
import pandas as pd


def recommend_action(risk_score: float) -> str:
    if risk_score < 0.3:
        return 'Automated SMS reminder'
    if risk_score < 0.6:
        return 'Reminder call'
    return 'Offer rescheduling / teleconsultation'


def build_interventions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    df = predictions_df.copy()
    df['recommended_action'] = df['risk_score'].apply(recommend_action)
    # ensure columns order
    return df[['appointment_id', 'predicted_status', 'risk_score', 'recommended_action']]


def write_interventions(interventions_df: pd.DataFrame, out_path='interventions.csv'):
    interventions_df.to_csv(out_path, index=False)
    print(f'Wrote interventions to {out_path}')


if __name__ == '__main__':
    # quick demo
    demo = pd.DataFrame({
        'appointment_id': ['A00001','A00002','A00003'],
        'predicted_status': ['No-show','Show','No-show'],
        'risk_score': [0.25, 0.48, 0.85]
    })
    out = build_interventions(demo)
    print(out)
