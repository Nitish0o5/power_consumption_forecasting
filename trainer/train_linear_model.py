import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Load data
data = pd.read_csv("data/training_data_raw_data_20250508_20_25.csv")

# Define input and output columns
input_columns = [col for col in data.columns if col.startswith('date_range_') or 
                col in ['time', 'consumed_power', 'time_sin', 'time_cos', 
                       'minute', 'second', 'minute_sin', 'minute_cos', 
                       'second_sin', 'second_cos']]
output_columns = ['white_goods', 'entertainment', 'air_conditioners', 'lighting', 'ev_charges','utility_appliances']


# Split features and targets
X = data[input_columns]
y = data[output_columns]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel evaluation:")
print(f"  - Mean Squared Error: {mse:.2f}")
print(f"  - R² Score: {r2:.2f}")

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to model/model.pkl")
