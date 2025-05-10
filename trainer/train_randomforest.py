import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Load preprocessed data
print("Loading preprocessed data...")
data = pd.read_csv("data/training_data_raw_data_20250508_20_25.csv")

# Define input and output columns
input_columns = [col for col in data.columns if col not in ['white_goods', 'entertainment', 'air_conditioners', 'lighting', 'ev_charges']]
output_columns = ['white_goods', 'entertainment', 'air_conditioners', 'lighting', 'ev_charges','utility_appliances']

# Prepare features and targets
X = data[input_columns]
y = data[output_columns]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
print("\nTraining Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate on test set
mse_rf = mean_squared_error(y_test, y_pred_rf, multioutput='uniform_average')
r2_rf = r2_score(y_test, y_pred_rf, multioutput='uniform_average')

print("\nRandom Forest Regressor Results (Test Set):")
print(f"   - MSE: {mse_rf:.2f}")
print(f"   - R²: {r2_rf:.2f}")

# Evaluate on training set
y_train_pred = rf_model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

print("\nRandom Forest Regressor Results (Training Set):")
print(f"   - MSE: {mse_train:.2f}")
print(f"   - R²: {r2_train:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': input_columns,
    'importance': rf_model.feature_importances_
})

# Save model
os.makedirs("model", exist_ok=True)
with open("model/random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("\nModel saved in model/")
