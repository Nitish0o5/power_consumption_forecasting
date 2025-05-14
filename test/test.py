import pandas as pd
import pickle
from datetime import datetime
import calendar
import numpy as np
import os
import sys

# Add trainer to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from trainer.time_feature_encoder import TimeFeatureEncoder

# Load model and encoders
with open("model/random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("model/time_encoder.pkl", "rb") as f:
    time_encoder = pickle.load(f)

# --- Function to convert date to date_range label ---
def get_date_range_label(date_obj):
    day = date_obj.day
    month = date_obj.strftime('%b').lower()
    days_in_month = calendar.monthrange(date_obj.year, date_obj.month)[1]
    range_size = days_in_month // 3
    if day <= range_size:
        return f"{month}_1"
    elif day <= range_size * 2:
        return f"{month}_2"
    else:
        return f"{month}_3"

# --- INPUT: user-provided raw data ---
input_data = {
    "date": "01:04:2024",          # DD:MM:YYYY
    "time": "14:00:00",            # HH:MM:SS
    "consumed_power": 3.719
}

# Step 1: Convert "date" to "date_range"
date_obj = datetime.strptime(input_data["date"], "%d:%m:%Y")
date_range_label = get_date_range_label(date_obj)

# Step 2: One-hot encode date_range
encoded_date = encoder.transform([[date_range_label]])
encoded_date_df = pd.DataFrame(encoded_date, columns=encoder.get_feature_names_out(["date_range"]))

# Step 3: Time features
time_features = time_encoder.transform(input_data["time"])  # dictionary
time_cols = ['time', 'time_sin', 'time_cos', 'minute', 'second',
             'minute_sin', 'minute_cos', 'second_sin', 'second_cos']

# Step 4: Combine all features
X_test = encoded_date_df.copy()
X_test["consumed_power"] = input_data["consumed_power"]
for col in time_cols:
    X_test[col] = time_features[col]

# Step 5: Final feature order as used in training
final_feature_order = [
    'date_range_apr_1', 'date_range_apr_2', 'date_range_apr_3',
    'date_range_aug_1', 'date_range_aug_2', 'date_range_aug_3',
    'date_range_dec_1', 'date_range_dec_2', 'date_range_dec_3',
    'date_range_feb_1', 'date_range_feb_2', 'date_range_feb_3',
    'date_range_jan_1', 'date_range_jan_2', 'date_range_jan_3',
    'date_range_jul_1', 'date_range_jul_2', 'date_range_jul_3',
    'date_range_jun_1', 'date_range_jun_2', 'date_range_jun_3',
    'date_range_mar_1', 'date_range_mar_2', 'date_range_mar_3',
    'date_range_may_1', 'date_range_may_2', 'date_range_may_3',
    'date_range_nov_1', 'date_range_nov_2', 'date_range_nov_3',
    'date_range_oct_1', 'date_range_oct_2', 'date_range_oct_3',
    'date_range_sep_1', 'date_range_sep_2', 'date_range_sep_3',
    'time', 'consumed_power',
    'time_sin', 'time_cos', 'minute', 'second',
    'minute_sin', 'minute_cos', 'second_sin', 'second_cos'
]

# Step 6: Fill any missing features with 0
for col in final_feature_order:
    if col not in X_test.columns:
        X_test[col] = 0

X_test = X_test[final_feature_order]

# Step 7: Predict
prediction = model.predict(X_test)

# Output
print("✅ Prediction Output:")
print(pd.DataFrame(prediction, columns=[
    'white_goods', 'entertainment', 'air_conditioners',
    'lighting', 'ev_charges', 'utility_appliances'
]))
