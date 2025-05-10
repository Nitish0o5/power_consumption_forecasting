import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import sys

def load_model_and_data():
    try:
        # Load the trained model
        print("Loading model from model/model.pkl...")                                        
        with open("model/model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Load test data
        print("Loading data from data/training_data_raw_data_20250506_15_30.csv...")
        data = pd.read_csv("data/training_data_raw_data_20250506_15_30.csv")
        
        # Get all date range columns first (they should be in alphabetical order)
        date_range_cols = sorted([col for col in data.columns if col.startswith('date_range_')])
        
        # Define input columns in the exact same order as training
        input_columns = date_range_cols + ['time', 'consumed_power', 'time_sin', 'time_cos', 
                                         'minute', 'second', 'minute_sin', 'minute_cos', 
                                         'second_sin', 'second_cos']
        
        # Add minute and second columns if they don't exist
        if 'minute' not in data.columns:
            print("Adding 'minute' column...")
            data['minute'] = data['time'].apply(lambda x: (x // 60) % 60)
            
        if 'second' not in data.columns:
            print("Adding 'second' column...")
            data['second'] = data['time'].apply(lambda x: x % 60)
        output_columns = ['white_goods', 'entertainment', 'air_conditioners', 'lighting', 'ev_charges']
        
        print(f"Input columns: {input_columns}")
        print(f"Output columns: {output_columns}")
        
        # Ensure all columns exist in the data
        missing_cols = [col for col in input_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")
        
        return model, data[input_columns], data[output_columns]
    except Exception as e:
        print(f"Error in load_model_and_data: {str(e)}", file=sys.stderr)
        raise

def evaluate_predictions(y_true, y_pred, categories):
    """Evaluate predictions for each category."""
    for i, category in enumerate(categories):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        print(f"\n{category}:")
        print(f"  - Mean Squared Error: {mse:.2f}")
        print(f"  - R² Score: {r2:.2f}")

def plot_predictions(y_true, y_pred, categories):
    """Plot actual vs predicted values for each category."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, category in enumerate(categories):
        if i < len(axes):
            ax = axes[i]
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
            ax.plot([y_true[:, i].min(), y_true[:, i].max()], 
                   [y_true[:, i].min(), y_true[:, i].max()], 
                   'r--', lw=2)
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{category}')
    
    plt.tight_layout()
    plt.savefig('test/prediction_results.png')
    plt.close()

def main():
    try:
        # Load model and data
        print("Loading model and data...")
        model, X, y = load_model_and_data()
        
        # Make predictions
        print("\nMaking predictions...")
        y_pred = model.predict(X)
        
        # Evaluate predictions
        print("\nEvaluation Results:")
        evaluate_predictions(y.values, y_pred, y.columns)
        
        # Plot results
        print("\nPlotting results...")
        plot_predictions(y.values, y_pred, y.columns)
        print("\nResults have been saved to test/prediction_results.png")
    except Exception as e:
        print(f"Error in main: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    main() 