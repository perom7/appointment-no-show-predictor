"""Main orchestrator for data generation, training, prediction and intervention output.

Usage:
    python main.py                      # uses/generates data/sample_data.csv
    python main.py --input path/to.csv  # run with your own CSV
    python main.py --interventions out.csv  # change interventions output filename
"""
import os
import argparse
from data.data_generator import generate_sample


def main():
    parser = argparse.ArgumentParser(description='Appointment No-Show Predictor pipeline')
    parser.add_argument('--input', type=str, default=None, help='Path to input CSV. If omitted, uses data/sample_data.csv (generated if missing).')
    parser.add_argument('--interventions', type=str, default='interventions.csv', help='Output CSV filename for interventions.')
    parser.add_argument('--model', type=str, choices=['baseline', 'balanced'], default='baseline', help='Which model artifact to train/use')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold for classifying No-show from risk score (0-1)')
    parser.add_argument('--predict-only', action='store_true', help='Skip training and only run prediction using existing model artifact')
    args = parser.parse_args()

    input_path = args.input

    # 1) Ensure sample data exists
    sample_path = os.path.join('data', 'sample_data.csv')
    if input_path is None:
        if not os.path.exists(sample_path):
            print('Generating sample data...')
            generate_sample(n=500)
        else:
            print(f'Found sample data at {sample_path}')
        input_path = sample_path
    else:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f'Input CSV not found: {input_path}')

    # 2) Train model (unless predict-only)
    from src.model_training import train_model
    artifact = None
    if not args.predict_only:
        print('\nTraining model...')
        # pick artifact path based on chosen model
        if args.model == 'balanced':
            model_path = 'models/model_balanced.joblib'
            artifact, info = train_model(input_path, model_out_path=model_path, class_weight='balanced')
        else:
            model_path = 'models/model.joblib'
            artifact, info = train_model(input_path, model_out_path=model_path, class_weight=None)
    else:
        print('\nPredict-only mode: skipping training')
        if args.model == 'balanced':
            model_path = 'models/model_balanced.joblib'
        else:
            model_path = 'models/model.joblib'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model artifact not found for predict-only: {model_path}. Run training first without --predict-only')
        from src.predictor import load_artifact
        artifact = load_artifact(model_path)

    # 3) Predict on upcoming appointments (here we reuse the sample set as a demo)
    print('\nPredicting risk scores...')
    from src.predictor import predict_for_df, load_artifact
    # ensure artifact is loaded
    if artifact is None:
        art = load_artifact(model_path)
    else:
        art = artifact
    preds = predict_for_df(input_path, art, threshold=args.threshold)

    # 4) Build interventions and write to CSV
    from src.intervention import build_interventions, write_interventions
    interventions = build_interventions(preds)
    write_interventions(interventions, out_path=args.interventions)

    # 5) Optional: scheduling demo
    try:
        from src.scheduling import spread_high_risk
        # merge on appointment_id so we have the 'time' column available for scheduling
        sample_cols = generate_sample_df_columns(input_path)
        merged = __import__('pandas').merge(preds, sample_cols, on='appointment_id', how='left')
        scheduled = spread_high_risk(merged, slot_column='time')
        # write scheduled sample for demonstration
        scheduled.to_csv('scheduled_demo.csv', index=False)
        print('Wrote scheduled demo to scheduled_demo.csv')
    except Exception as e:
        # non-fatal; scheduling is optional and demo helper may not be available
        print('Scheduling demo skipped (optional). Reason:', str(e))


def generate_sample_df_columns(path):
    # helper to return minimal columns needed for scheduling demo
    import pandas as pd
    df = pd.read_csv(path)
    return df[['appointment_id','time']]


if __name__ == '__main__':
    main()
