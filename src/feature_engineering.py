"""Feature engineering utilities.

Functions:
- load_and_process(df_or_path): load CSV or DataFrame, create derived features, handle missing, encode categoricals,
  and return X, y, and the full processed DataFrame.
"""
from typing import Tuple
import pandas as pd
import numpy as np
from datetime import datetime


def load_data(path_or_df):
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    # parse dates
    df = df.copy()
    df['appointment_date'] = pd.to_datetime(df['appointment_date'])
    df['booking_date'] = pd.to_datetime(df['booking_date'])
    df['lead_days'] = (df['appointment_date'] - df['booking_date']).dt.days
    df['appointment_dow'] = df['appointment_date'].dt.dayofweek  # 0=Mon
    df['is_weekend'] = df['appointment_dow'].isin([5,6]).astype(int)
    return df


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X, y, processed_df

    - Handles missing values
    - Creates derived features
    - Encodes simple categoricals with one-hot
    """
    df = load_data(df)
    df = add_temporal_features(df)

    # target
    if 'no_show' in df.columns:
        y = df['no_show']
    else:
        y = None

    # Fill missing gender with 'U'
    df['gender'] = df['gender'].fillna('U')

    # Fill weather
    df['weather'] = df.get('weather', pd.Series(['Unknown']*len(df))).fillna('Unknown')

    # Age: fill with median
    df['age'] = df['age'].fillna(df['age'].median())

    # previous_no_shows: fill 0
    df['previous_no_shows'] = df['previous_no_shows'].fillna(0)

    # Extract hour from time string
    def hour_from_time(t):
        try:
            return int(str(t).split(':')[0])
        except Exception:
            return 10

    df['appointment_hour'] = df['time'].apply(hour_from_time)

    # Derived: previous no-show rate per patient if multiple rows exist
    # Simple group-based feature
    if 'patient_id' in df.columns:
        pn = df.groupby('patient_id')['previous_no_shows'].sum().rename('patient_total_prev_no_shows')
        df = df.join(pn, on='patient_id')
    else:
        df['patient_total_prev_no_shows'] = df['previous_no_shows']

    # One-hot encode categorical variables: gender, department, appointment_type, weather
    cat_cols = ['gender', 'department', 'appointment_type', 'weather']
    df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], dummy_na=False)

    # Features to use
    feature_cols = [
        'age', 'lead_days', 'is_weekend', 'appointment_hour', 'previous_no_shows', 'patient_total_prev_no_shows'
    ]
    # add any generated dummies
    feature_cols += [c for c in df.columns if any(prefix in c for prefix in ['gender_', 'department_', 'appointment_type_', 'weather_'])]

    X = df[feature_cols].fillna(0)

    return X, y, df


if __name__ == '__main__':
    # quick smoke test
    from data.data_generator import generate_sample
    generate_sample(n=50)
    X, y, df = preprocess('data/sample_data.csv')
    print('Processed', X.shape)
