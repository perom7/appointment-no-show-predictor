# Appointment No-Show Predictor

Project overview
----------------
This project is a modular machine learning prototype that predicts which patients are likely to miss
their scheduled appointments (no-shows) and recommends targeted interventions to reduce missed
appointments. It focuses on the core ML pipeline: data preparation, feature engineering, model
training, prediction, and intervention recommendation. An optional scheduling helper demonstrates
how high-risk appointments could be spread across time slots.

Key components
--------------
- data/data_generator.py — synthetic appointment dataset generator (writes `data/sample_data.csv`).
- src/feature_engineering.py — preprocessing and derived feature creation (lead time, weekend flag,
	appointment hour, aggregated previous no-shows, categorical encoding).
- src/model_training.py — trains a Random Forest model, evaluates on a test split, prints metrics
	(accuracy, precision, recall, F1) and feature importances, and saves the model artifact.
- src/predictor.py — loads a saved model artifact, computes risk probabilities, and returns
	predictions.
- src/intervention.py — maps risk scores to recommended actions and writes an interventions CSV
	with `appointment_id`, `predicted_status`, `risk_score`, and `recommended_action`.
- src/scheduling.py — optional greedy scheduler to spread high-risk appointments across time slots.
- main.py — orchestrator that runs generation, training, prediction, and writes outputs.

Data and expected inputs
------------------------
The pipeline accepts an appointments CSV. Typical columns used by the code:
`appointment_id`, `patient_id`, `appointment_date`, `booking_date`, `time`, `age`, `gender`,
`department`, `appointment_type`, `previous_no_shows`, and optional `weather`.

Outputs
-------
- Interventions CSV (default `interventions.csv`) containing `appointment_id`, `predicted_status`,
	`risk_score`, and `recommended_action`.
- Model artifacts under `models/` (e.g., `model.joblib`, `model_balanced.joblib`).

Design notes
------------
- The model uses a Random Forest as a sensible baseline. An `experiments` script demonstrates
	class-weight balancing and threshold sweeps to trade off precision vs recall.
- Intervention rules are simple, rule-based mappings from probability to action (SMS, call,
	reschedule/teleconsultation). Thresholds are configurable in the code/CLI.

Limitations
-----------
- The included dataset is synthetic and intended for demonstration only. Real-world deployment
	requires validated, privacy-compliant data, additional model validation, and operational testing.
- The prototype focuses on the ML pipeline; production concerns (monitoring, logging, secure data
	handling) are outside its current scope.
