import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import pickle
import os

# Load data
data = pd.read_csv("data/training_data_raw_data_20250508_20_25.csv")
data = data.dropna(subset=['time', 'consumed_power'])

# Convert time column to seconds (assuming proper HH:MM:SS format)
def parse_time_string(t):
    try:
        dt = pd.to_datetime(t, format='%H:%M:%S')
        return dt.hour * 3600 + dt.minute * 60 + dt.second
    except:
        return np.nan

data['time'] = data['time'].apply(parse_time_string)
data.dropna(subset=['time'], inplace=True)

# Add cyclical time features
data['minute'] = (data['time'] % 3600) // 60
data['second'] = data['time'] % 60
data['time_sin'] = np.sin(2 * np.pi * data['time'] / (24 * 3600))
data['time_cos'] = np.cos(2 * np.pi * data['time'] / (24 * 3600))
data['minute_sin'] = np.sin(2 * np.pi * data['minute'] / 60)
data['minute_cos'] = np.cos(2 * np.pi * data['minute'] / 60)
data['second_sin'] = np.sin(2 * np.pi * data['second'] / 60)
data['second_cos'] = np.cos(2 * np.pi * data['second'] / 60)

# Select features and targets
features = ['time', 'time_sin', 'time_cos', 'minute_sin', 'minute_cos', 'second_sin', 'second_cos', 'consumed_power']
targets = ['white_goods', 'entertainment', 'air_conditioners', 'lighting', 'ev_charges', 'utility_appliances']

# Normalize features and targets
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_all = feature_scaler.fit_transform(data[features])
y_all = target_scaler.fit_transform(data[targets])

# Create sequences
window_size = 10
X_seq, y_seq = [], []
for i in range(len(X_all) - window_size):
    X_seq.append(X_all[i:i+window_size])
    y_seq.append(y_all[i+window_size])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(X_seq.shape[1], X_seq.shape[2]), return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(32, activation='tanh'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(targets), activation='linear')
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    ],
    verbose=1
)

# Evaluate model
y_pred = model.predict(X_test)
y_pred_inv = target_scaler.inverse_transform(y_pred)
y_test_inv = target_scaler.inverse_transform(y_test)

print("\nPerformance by Appliance Category:")
for i, name in enumerate(targets):
    print(f"\n{name}:")
    print(f"  R² Score: {r2_score(y_test_inv[:, i], y_pred_inv[:, i]):.4f}")
    print(f"  MSE: {mean_squared_error(y_test_inv[:, i], y_pred_inv[:, i]):.4f}")
    print(f"  MAE: {mean_absolute_error(y_test_inv[:, i], y_pred_inv[:, i]):.4f}")

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/lstm_model.keras")
with open("model/feature_scaler.pkl", "wb") as f:
    pickle.dump(feature_scaler, f)
with open("model/target_scaler.pkl", "wb") as f:
    pickle.dump(target_scaler, f)

print("\nModel and scalers saved to 'model/'")
