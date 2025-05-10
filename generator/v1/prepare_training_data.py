import pandas as pd

def prepare_training_data(file_path, columns_to_use, output_path):
    
    try:
        
        df = pd.read_csv(file_path)

        
        missing = [col for col in columns_to_use if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

        
        training_data = df[columns_to_use].copy()

        
        training_data.to_csv(output_path, index=False)
        print(f" Training data saved to {output_path}")
        return output_path

    except Exception as e:
        print(f" Error: {e}")
        return None


if __name__ == "__main__":
    file_path = "data/raw_data_20250505_12_57.csv"
    output_file = "data/training_data_raw_data_20250505_12_57.csv"
    selected_columns = [
        "date_range","time","consumed_power", "white_goods","entertainment","air_conditioners","lighting","ev_charges" 
    ]
    prepare_training_data(file_path, selected_columns, output_file)
