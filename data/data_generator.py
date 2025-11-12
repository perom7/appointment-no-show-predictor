"""Synthetic appointment data generator.

Creates a CSV of appointment records with realistic-ish fields:
- appointment_id, patient_id, appointment_date, booking_date, time, age, gender,
  department, appointment_type, previous_no_shows, weather (optional), no_show (target)

Run this module to generate `data/sample_data.csv`.
"""
from datetime import datetime, timedelta
import random
import csv
import os

OUT_PATH = os.path.join(os.path.dirname(__file__), "sample_data.csv")

def generate_sample(n=500, seed=42, out_path=OUT_PATH):
    random.seed(seed)
    headers = [
        "appointment_id", "patient_id", "appointment_date", "booking_date", "time",
        "age", "gender", "department", "appointment_type", "previous_no_shows", "weather", "no_show"
    ]

    start_date = datetime.today()

    departments = ["General", "Cardiology", "Dermatology", "Pediatrics", "Orthopedics"]
    appointment_types = ["Follow-up", "New", "Routine", "Urgent"]
    weather_states = ["Sunny", "Rainy", "Snow", "Cloudy"]

    rows = []
    for i in range(1, n+1):
        appointment_date = start_date + timedelta(days=random.randint(1, 60))
        booking_date = appointment_date - timedelta(days=random.randint(0, 90))
        time_slot = f"{random.randint(8, 17)}:{random.choice(["00","15","30","45"])}"
        age = random.randint(0, 90)
        gender = random.choice(["M", "F"]) if random.random() > 0.05 else None  # some missing
        dept = random.choice(departments)
        appt_type = random.choice(appointment_types)
        prev_no_shows = random.choices([0,1,2,3,4], weights=[60,20,10,6,4])[0]
        weather = random.choice(weather_states) if random.random() > 0.2 else None

        # Simple heuristic for no-show: younger/older, previous no-shows, rainy weather -> higher chance
        base_prob = 0.08
        if age < 25: base_prob += 0.05
        if age > 75: base_prob += 0.06
        base_prob += prev_no_shows * 0.10
        if weather == "Rainy": base_prob += 0.05
        # longer lead time increases forgetfulness a bit
        lead_days = (appointment_date - booking_date).days
        if lead_days > 30: base_prob += 0.03

        no_show = 1 if random.random() < min(base_prob, 0.9) else 0

        rows.append([
            f"A{i:06d}", f"P{random.randint(1,2000):05d}",
            appointment_date.strftime("%Y-%m-%d"), booking_date.strftime("%Y-%m-%d"), time_slot,
            age, gender, dept, appt_type, prev_no_shows, weather, no_show
        ])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    generate_sample(n=500)
