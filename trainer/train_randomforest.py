import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Load preprocessed data
print("🔄 Loading preprocessed data...")
data = pd.read_csv("data/training_data_raw_data_20250508_20_25.csv")

# Define input and output columns
target_columns = ['white_goods', 'entertainment', 'air_conditioners', 'lighting', 'ev_charges', 'utility_appliances']
input_columns = [col for col in data.columns if col not in target_columns]

# Prepare features and targets
X = data[input_columns]
y = data[target_columns]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
print("\n🌲 Training Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_test = rf_model.predict(X_test)
y_pred_train = rf_model.predict(X_train)

# Evaluate test set
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

# Evaluate training set
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

print("\n📊 Random Forest Regressor Results (Test Set):")
print(f"   - MSE: {mse_test:.4f}")
print(f"   - R² : {r2_test:.4f}")

print("\n📈 Random Forest Regressor Results (Training Set):")
print(f"   - MSE: {mse_train:.4f}")
print(f"   - R² : {r2_train:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': input_columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n🔥 Top Features:")
print(feature_importance.head(10))

# Optionally export feature importance
feature_importance.to_csv("model/feature_importance.csv", index=False)

# Save model
os.makedirs("model", exist_ok=True)
with open("model/random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("\n✅ Model and feature importances saved to 'model/'")
