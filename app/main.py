from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import dill as pickle
import sys
import trainer.time_feature_encoder as tfe
from datetime import datetime
import calendar
import os
import sys

# Add trainer folder to path (for custom encoder)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from trainer.time_feature_encoder import TimeFeatureEncoder

# Load model and encoders
with open("model/random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
original_main = sys.modules.get('__main__')
sys.modules['__main__'] = tfe
with open("model/time_encoder.pkl", "rb") as f:
    time_encoder = pickle.load(f)
if original_main is not None:
    sys.modules['__main__'] = original_main
else:
    del sys.modules['__main__']

# Final feature order
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

# Define input schema
class PredictionInput(BaseModel):
    date: str  # Format: DD:MM:YYYY
    time: str  # Format: HH:MM:SS
    consumed_power: float

# Create FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(input_data: PredictionInput):
    # --- Process Input ---
    date_obj = datetime.strptime(input_data.date, "%d:%m:%Y")
    date_range_label = get_date_range_label(date_obj)

    encoded_date = encoder.transform([[date_range_label]])
    encoded_date_df = pd.DataFrame(encoded_date, columns=encoder.get_feature_names_out(["date_range"]))

    time_features = time_encoder.transform(input_data.time)
    X_test = encoded_date_df.copy()
    X_test["consumed_power"] = input_data.consumed_power

    for col in time_features:
        X_test[col] = time_features[col]

    for col in final_feature_order:
        if col not in X_test.columns:
            X_test[col] = 0

    X_test = X_test[final_feature_order]

    # --- Predict ---
    prediction = model.predict(X_test)[0]

    return {
        "white_goods": prediction[0],
        "entertainment": prediction[1],
        "air_conditioners": prediction[2],
        "lighting": prediction[3],
        "ev_charges": prediction[4],
        "utility_appliances": prediction[5],
    }

