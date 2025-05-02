# training/train_model.py
import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load training data
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
