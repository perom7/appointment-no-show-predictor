"""Optional scheduling optimizer.

Provides a simple greedy method to spread high-risk appointments across available time slots
to reduce clustering of high-risk cases.
"""
import pandas as pd
from typing import List


def spread_high_risk(df: pd.DataFrame, slot_column: str = 'time', risk_column: str = 'risk_score') -> pd.DataFrame:
    """Return a DataFrame with the same rows but reassigned `slot_column` values in a round-robin
    manner so that high-risk rows are spread across slots.

    This is a simple heuristic example and assumes `slot_column` contains a small set of time labels.
    """
    df = df.copy()
    slots: List[str] = df[slot_column].unique().tolist()
    slots_sorted = sorted(slots)
    # sort by risk descending, then assign slots round-robin
    df_sorted = df.sort_values(by=risk_column, ascending=False).reset_index(drop=True)
    assigned_slots = [slots_sorted[i % len(slots_sorted)] for i in range(len(df_sorted))]
    df_sorted[slot_column] = assigned_slots
    # restore original index ordering
    return df_sorted


if __name__ == '__main__':
    import pandas as pd
    demo = pd.DataFrame({
        'appointment_id': [f'A{i:03d}' for i in range(12)],
        'time': ['09:00','09:15','09:30','09:45'] * 3,
        'risk_score': [0.9,0.8,0.1,0.2,0.7,0.05,0.4,0.6,0.2,0.9,0.12,0.55]
    })
    print(spread_high_risk(demo))
