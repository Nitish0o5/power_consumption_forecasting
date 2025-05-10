import numpy as np
import pickle
import os
from datetime import datetime

class TimeFeatureEncoder:
    def __init__(self):
        self.time_format = '%H:%M:%S'
    
    def convert_time_to_seconds(self, time_str):
        """Convert time string (HH:MM:SS) to seconds."""
        time_obj = datetime.strptime(time_str, self.time_format)
        return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    
    def add_cyclical_features(self, time_seconds):
        """Convert time in seconds to cyclical features."""
        # Hourly cycle (24 hours)
        time_normalized = 2 * np.pi * time_seconds / (24 * 3600)
        time_sin = np.sin(time_normalized)
        time_cos = np.cos(time_normalized)
        
        # Minute cycle (60 minutes)
        minute = (time_seconds // 60) % 60
        minute_normalized = 2 * np.pi * minute / 60
        minute_sin = np.sin(minute_normalized)
        minute_cos = np.cos(minute_normalized)
        
        # Second cycle (60 seconds)
        second = time_seconds % 60
        second_normalized = 2 * np.pi * second / 60
        second_sin = np.sin(second_normalized)
        second_cos = np.cos(second_normalized)
        
        return {
            'time': time_seconds,
            'time_sin': time_sin,
            'time_cos': time_cos,
            'minute': minute,
            'second': second,
            'minute_sin': minute_sin,
            'minute_cos': minute_cos,
            'second_sin': second_sin,
            'second_cos': second_cos
        }
    
    def transform(self, time_str):
        """Transform a time string to all required features."""
        time_seconds = self.convert_time_to_seconds(time_str)
        return self.add_cyclical_features(time_seconds)
    
    def transform_batch(self, time_strings):
        """Transform a list of time strings to features."""
        features = []
        for time_str in time_strings:
            features.append(self.transform(time_str))
        return features

def main():
    # Create encoder
    encoder = TimeFeatureEncoder()
    
    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    
    # Save encoder
    with open("model/time_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    
    print("Time feature encoder saved to model/time_encoder.pkl")
    
    # Test the encoder
    test_time = "14:30:45"
    features = encoder.transform(test_time)
    print("\nTest conversion:")
    print(f"Input time: {test_time}")
    print("Output features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main() 