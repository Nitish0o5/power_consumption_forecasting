import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def add_cyclical_features(df, col, period):
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
    return df

def prepare_training_data(file_path, columns_to_use, output_path):
    try:
        # Load data
        print("Loading data...")
        df = pd.read_csv(file_path)
        
        # Check for missing columns
        missing = [col for col in columns_to_use if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")
        
        # Get the data
        training_data = df[columns_to_use].copy()
        
        # Convert time to seconds
        print("Converting time to seconds...")
        time_dt = pd.to_datetime(training_data['time'], format='%H:%M:%S')
        training_data['time'] = time_dt.dt.hour * 3600 + time_dt.dt.minute * 60 + time_dt.dt.second
        
        # Add cyclical time features
        print("Adding cyclical features...")
        training_data = add_cyclical_features(training_data, 'time', 24 * 3600)
        training_data['minute'] = (training_data['time'] % 3600) // 60
        training_data['second'] = training_data['time'] % 60
        training_data = add_cyclical_features(training_data, 'minute', 60)
        training_data = add_cyclical_features(training_data, 'second', 60)
        
        # Label encode date_range
        print("Label encoding date_range...")
        label_encoder = LabelEncoder()
        training_data['date_range'] = label_encoder.fit_transform(training_data['date_range'])
        
        # Save to CSV
        print(f"Saving processed data to {output_path}")
        training_data.to_csv(output_path, index=False)
        print("Data preparation completed successfully!")
        
        return output_path

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    file_path = "data/raw_data_20250506_15_30.csv"
    output_file = "data/training_data_raw_data_20250506_15_30_sample.csv"
    selected_columns = [
        "date_range", "time", "consumed_power", 
        "white_goods", "entertainment", "air_conditioners", 
        "lighting", "ev_charges"
    ]
    prepare_training_data(file_path, selected_columns, output_file)