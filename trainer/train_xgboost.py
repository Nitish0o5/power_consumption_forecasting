import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os

def load_data(file_path):
    """Load data and generate time features."""
    df = pd.read_csv(file_path)
    
    # Create minute and second from 'time' if needed
    df['minute'] = df['time'] // 60 % 60
    df['second'] = df['time'] % 60

    # Create sine/cosine features
    df['time_sin'] = np.sin(2 * np.pi * df['time'] / (24 * 3600))
    df['time_cos'] = np.cos(2 * np.pi * df['time'] / (24 * 3600))
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['second_sin'] = np.sin(2 * np.pi * df['second'] / 60)
    df['second_cos'] = np.cos(2 * np.pi * df['second'] / 60)
    
    return df

def train_model(data):
    input_cols = [col for col in data.columns if col.startswith('date_range_') or col in [
        'time', 'consumed_power', 'time_sin', 'time_cos', 
        'minute_sin', 'minute_cos', 'second_sin', 'second_cos']]
    
    output_cols = ['white_goods', 'entertainment', 'air_conditioners', 'lighting', 'ev_charges','utility_appliances']
    
    X = data[input_cols]
    y = data[output_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Calculate RMSE manually
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.2f}")
    
    return model

def save_model(model, path="model/xgboost_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    data = load_data("data/training_data_raw_data_20250508_20_25.csv")
    model = train_model(data)
    save_model(model)