import pandas as pd
import pickle
from datetime import datetime
import sys
import os

# Add the trainer directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the TimeFeatureEncoder class
from trainer.time_feature_encoder import TimeFeatureEncoder

# Load saved model and encoder
with open("model/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("model/time_encoder.pkl", "rb") as f:
    time_encoder = pickle.load(f)

# --- EXAMPLE INPUT ---
# You can change this to test different values
input_data = {
    "date_range": "2025-05-01",
    "time": "13:45:30",  # HH:MM:SS
    "consumed_power": 220
}

# Convert time to cyclical features using the time encoder
time_features = time_encoder.transform(input_data["time"])

# One-hot encode 'date_range'
encoded_date = encoder.transform([[input_data["date_range"]]])

# Construct final input DataFrame with features in the correct order
X_test = pd.DataFrame(encoded_date, columns=encoder.get_feature_names_out(["date_range"]))

# Add consumed power first as it's likely the first feature in training
X_test["consumed_power"] = input_data["consumed_power"]

# Add time features in the correct order
feature_order = ['time', 'time_sin', 'time_cos', 'minute', 'second', 
                'minute_sin', 'minute_cos', 'second_sin', 'second_cos']
for feature in feature_order:
    X_test[feature] = time_features[feature]

# Ensure column order matches training data
all_features = encoder.get_feature_names_out(["date_range"]).tolist() +["consumed_power"]+ feature_order
X_test = X_test[all_features]
print(X_test)
# Predict
prediction = model.predict(X_test)

# Show prediction result
print("✅ Prediction Output:")
print(prediction)
