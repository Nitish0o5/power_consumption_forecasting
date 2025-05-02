import pandas as pd

def prepare_training_data(file_path, columns_to_use, output_path):
    
    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        # Check if all requested columns exist
        missing = [col for col in columns_to_use if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

        # Select the desired columns
        training_data = df[columns_to_use].copy()

        # Save the output
        training_data.to_csv(output_path, index=False)
        print(f" Training data saved to {output_path}")
        return output_path

    except Exception as e:
        print(f" Error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    file_path = "data/raw_data_20250501_17_52.csv"
    output_file = "data/training_data_raw_data_20250501_17_52.csv"
    selected_columns = [
        "meter_reading", "consumed_power", "white_goods", 
    ]
    prepare_training_data(file_path, selected_columns, output_file)
