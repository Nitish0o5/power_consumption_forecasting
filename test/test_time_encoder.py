import pickle
import pandas as pd
import numpy as np
import sys
import os

# Add the trainer directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the TimeFeatureEncoder class
from trainer.time_feature_encoder import TimeFeatureEncoder

def load_time_encoder():
    """Load the time feature encoder."""
    with open("model/time_encoder.pkl", "rb") as f:
        return pickle.load(f)

def test_single_time():
    """Test the encoder with a single time value."""
    encoder = load_time_encoder()
    
    # Test with a specific time
    test_time = "14:30:45"
    features = encoder.transform(test_time)
    
    print(f"\nTesting single time conversion:")
    print(f"Input time: {test_time}")
    print("Output features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")

def test_batch_conversion():
    """Test the encoder with multiple time values."""
    encoder = load_time_encoder()
    
    # Test with multiple times
    test_times = ["09:15:30", "14:30:45", "23:59:59"]
    features_list = encoder.transform_batch(test_times)
    
    print(f"\nTesting batch conversion:")
    for time_str, features in zip(test_times, features_list):
        print(f"\nInput time: {time_str}")
        print("Output features:")
        for key, value in features.items():
            print(f"  {key}: {value:.4f}")

def test_with_dataframe():
    """Test the encoder with a pandas DataFrame."""
    encoder = load_time_encoder()
    
    # Create a sample DataFrame with time column
    df = pd.DataFrame({
        'time': ["09:15:30", "14:30:45", "23:59:59"]
    })
    
    # Convert times to features
    features_list = encoder.transform_batch(df['time'])
    
    # Create a new DataFrame with all features
    features_df = pd.DataFrame(features_list)
    
    print("\nTesting with DataFrame:")
    print("\nOriginal DataFrame:")
    print(df)
    print("\nConverted Features DataFrame:")
    print(features_df)

def main():
    try:
        # Test single time conversion
        test_single_time()
        
        # Test batch conversion
        test_batch_conversion()
        
        # Test with DataFrame
        test_with_dataframe()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 