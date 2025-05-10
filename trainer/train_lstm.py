import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import r2_score

# Model configuration
additional_metrics = ['accuracy']
batch_size = 128
embedding_output_dims = 15
loss_function = BinaryCrossentropy()
max_sequence_length = 300
num_distinct_words = 5000
number_of_epochs = 50
optimizer = Adam()
validation_split = 0.20
verbosity_mode = 1

# Disable eager execution
# tf.compat.v1.disable_eager_execution()

# Load dataset from a CSV file
data = pd.read_csv('data/training_data_raw_data_20250508_20_25.csv')  # Update with your dataset path

input_columns = [col for col in data.columns if col.startswith('date_range_') or 
                col in ['time', 'consumed_power', 'time_sin', 'time_cos', 
                       'minute', 'second', 'minute_sin', 'minute_cos', 
                       'second_sin', 'second_cos']]
output_columns = ['white_goods', 'entertainment', 'air_conditioners', 'lighting', 'ev_charges']


x_data = data[input_columns].values  # Replace 'text_column' with the actual column name
y_data = data[output_columns].values  # Replace 'label_column' with the actual label column name

# Preprocess the text data (tokenization, etc.)
# Assuming you have a function to preprocess your text data


# Reshape x_data to 3D array (samples, max_sequence_length, num_features)
num_samples = x_data.shape[0] // max_sequence_length
num_features = x_data.shape[1]

# Truncate x_data to have full sequences only
x_data_trimmed = x_data[:num_samples * max_sequence_length, :]

# Truncate y_data to have full sequences only
y_data_trimmed = y_data[:num_samples * max_sequence_length, :]

# Aggregate y_data_trimmed to shape (num_samples, 5) by taking mean over each sequence
import numpy as np
y_data_aggregated = y_data_trimmed.reshape(num_samples, max_sequence_length, y_data.shape[1]).mean(axis=1)

# Reshape to (num_samples, max_sequence_length, num_features)
padded_inputs = x_data_trimmed.reshape((num_samples, max_sequence_length, num_features))

# Define the Keras model
model = Sequential()
model.add(LSTM(10, input_shape=(max_sequence_length, len(input_columns))))
model.add(Dense(5, activation='sigmoid'))

# Compile the model
model.compile(optimizer=optimizer, loss=loss_function, metrics=additional_metrics)

# Give a summary
model.summary()

# Train the model
history = model.fit(padded_inputs, y_data_aggregated, batch_size=batch_size, epochs=number_of_epochs, verbose=verbosity_mode, validation_split=validation_split)

# Test the model after training
test_results = model.evaluate(padded_inputs, y_data_aggregated, verbose=False)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {100 * test_results[1]}%')

# Predict on the training data
y_pred = model.predict(padded_inputs)

# Calculate R2 score
r2 = r2_score(y_data_aggregated, y_pred)
print(f'R2 score: {r2}')
