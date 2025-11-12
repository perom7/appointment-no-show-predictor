# Appointment No-Show Predictor

Predicts patient appointment no-shows and recommends interventions to reduce missed appointments.

Features
- Synthetic data generator for appointment data
- Feature engineering (lead time, day of week, previous no-show rate, etc.)
- Model training (Random Forest) with evaluation metrics and feature importances
- Prediction + risk score output and recommended intervention rules
- Optional simple scheduling optimizer to spread high-risk appointments

Quick start
1. Create a Python 3.9+ virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the main script (this will generate sample data, train a model, and write interventions):

```powershell
python main.py
```

Files
- `data/data_generator.py`: Generates synthetic appointment dataset and writes `data/sample_data.csv`.
- `src/feature_engineering.py`: Build derived features and preprocessing.
- `src/model_training.py`: Train, evaluate, and save model to `models/model.joblib`.
- `src/predictor.py`: Load model and produce predictions and risk scores.
- `src/intervention.py`: Map risk scores to recommended actions and write output CSV.
- `src/scheduling.py`: Optional simple scheduler to spread high-risk appointments.
- `main.py`: Orchestrator.

Input/Output
- Input: CSV with appointment records (sample generation included). Key fields: `appointment_id`, `patient_id`, `appointment_date`, `booking_date`, `age`, `gender`, `department`, `appointment_type`, `previous_no_shows`, `time`.
- Output: `interventions.csv` with columns `appointment_id`, `predicted_status`, `risk_score`, `recommended_action`.

Notes
- The repository does not push to GitHub automatically. Use the commands below to create a public repo and push your code.

Git push example (run locally):

```powershell
git init
git add .
git commit -m "Initial commit: appointment no-show predictor"
git branch -M main
# create a repo on GitHub and then:
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
