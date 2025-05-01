import os
import json

def read_house_json_files(folder_path):
    # List all files in the given folder
    for filename in os.listdir(folder_path):
        if filename.startswith("house") and filename.endswith(".json"):
            full_path = os.path.join(folder_path, filename)
            try:
                with open(full_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    print(f"Contents of {filename}:")
                    print(data)
            except Exception as e:
                print(f"Failed to read {filename}: {e}")

# Example usage
folder_path = "configuration"  # Replace with your folder path
read_house_json_files(folder_path)
